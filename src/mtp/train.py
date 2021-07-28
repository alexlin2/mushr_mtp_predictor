import re
from functools import partial
from itertools import product, permutations
from operator import itemgetter
from time import time
from typing import Dict, Any, List, Tuple

import torch
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import MSELoss, CrossEntropyLoss
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtp.config import TrainingConfig
from mtp.networks import GraphNetWrapper
from mtp.utils.logging import get_logger

logger = get_logger(__name__)


def onehot_from_index(
        index_vector: Tensor,
        num_cls: int) -> Tensor:
    onehot = torch.zeros((index_vector.numel(), num_cls), dtype=torch.float32, device=index_vector.device)
    for i in range(index_vector.numel()):
        onehot[i, index_vector[i]] = 1.
    return onehot


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def winding_number_combination(num_agent: int) -> Tuple[List[Tuple[int, ...]], ...]:
    edge_indices = list(permutations(range(num_agent), 2))
    winding_candidates = list(product(range(2), repeat=len(edge_indices)))
    constraints = []
    for i, m in enumerate(edge_indices):
        for j, n in enumerate(edge_indices):
            if m[0] == n[1] and m[1] == n[0]:
                constraints.append((i, j))
    constraints = list(set(tuple(sorted(c)) for c in constraints))
    winding_combinations = list(filter(lambda x: all(x[i] == x[j] for i, j in constraints), winding_candidates))
    unique_constraints = list(map(itemgetter(0), sorted(constraints)))
    unique_edges = [edge_indices[u] for u in unique_constraints]
    unique_winding_combinations = [tuple(w[u] for u in unique_constraints) for w in winding_combinations]
    return unique_constraints, unique_winding_combinations, winding_combinations


def vae_loss(loss_func, beta, prd_value, tar_value, mu, logvar):
    bce = loss_func(prd_value, tar_value)
    kld = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / len(mu)
    return bce + beta * kld


def prep_data(ref_trajectory, item_index: int, unique_samples, dim_goal: int):
    """
    Prepare the data for trajectory prediction from the samples
    :param ref_trajectory: torch.float32, (batch_size, num_agents, num_ref_frames, dim_states)
    :param item_index: index in the batch
    :param unique_samples: torch.float32, (num_unique_samples, num_agents * (dim_goal + dim_winding))
    :param dim_goal: int, dimension of one-hot goal vector
    :return:
        :curr_input: torch.float32, (num_unique_samples, num_agents, num_ref_frames, dim_states)
        :pred_winding: torch.float32, (num_unique_samples, num_agents, dim_winding)
        :pred_goal: torch.float32, (num_unique_samples, num_agents, dim_goal)
    """
    num_unique_samples = unique_samples.shape[0]
    num_agents = unique_samples.shape[1]
    unique_samples = unique_samples.view(num_unique_samples, num_agents, -1)
    pred_goal = unique_samples[..., :dim_goal]  # (num_unique_samples, num_agents, dim_goal)
    pred_winding = unique_samples[..., dim_goal:]  # (num_unique_samples, num_agents, dim_winding)
    curr_input = ref_trajectory[item_index, ...].squeeze().unsqueeze(0). \
        expand(num_unique_samples, -1, -1,
               -1)  # (num_unique_samples, num_agents, num_ref_frames, dim_states)
    return curr_input, pred_winding, pred_goal


class Trainer:
    def __init__(
            self,
            model_wrapper: GraphNetWrapper,
            modes: List[str],
            dataloader: Dict[str, DataLoader],
            run_every: Dict[str, int],
            config: TrainingConfig):
        self.model_wrapper = model_wrapper
        self.config = config

        self.num_epoch = 0
        self.num_iter = 0
        self.optim_winding = torch.optim.Adam(self.winding_model.parameters(), lr=self.config.lr)
        self.optim_trajectory = torch.optim.Adam(self.trajectory_model.parameters(), lr=self.config.lr)
        self.losses = {
            'node': MSELoss(),
            'winding': partial(vae_loss, CrossEntropyLoss(), self.beta),
            'goal': partial(vae_loss, CrossEntropyLoss(), self.beta),
            'edge': MSELoss()
        }
        _dirs = {'train': self.config.train_tb_dir,
                 'test': self.config.test_tb_dir}
        self.tb_writer = {m: SummaryWriter(str(d), flush_secs=self.config.flush_secs) for m, d in _dirs.items()}

        self.modes = modes
        self.dataloader = dataloader
        self.data_iter = {m: iter(self.dataloader[m]) for m in self.modes}
        self.run_every = run_every
        self.finished = False
        self.meter_dict = dict()

        self.load()

    @property
    def winding_model(self):
        return self.model_wrapper.winding_model

    @property
    def trajectory_model(self):
        return self.model_wrapper.trajectory_model

    def retrieve_data(self) -> Dict[str, Any]:
        data = {}
        try:
            for m in self.modes:
                if self.num_iter % self.run_every[m] == 0:
                    data[m] = next(self.data_iter[m])
        except StopIteration:
            self.num_epoch += 1
            if self.num_iter >= self.max_iter:
                self.finished = True
                return {}
            for m in self.modes:
                self.data_iter[m] = iter(self.dataloader[m])
                if self.num_iter % self.run_every[m] == 0:
                    data[m] = next(self.data_iter[m])
        return data

    def initialize_meter_dict(self):
        self.meter_dict = {m: dict() for m in self.modes}
        meter_keys = ['batch_time', 'data_time', 'total_loss', 'winding_loss', 'goal_loss', 'node_loss', 'edge_loss']
        for m in self.modes:
            for key in meter_keys:
                self.meter_dict[m][key] = AverageMeter()

    @property
    def beta(self):
        return self.config.beta

    @property
    def max_iter(self):
        return self.config.max_iter

    @property
    def device(self):
        return self.model_wrapper.device

    @property
    def u_dim(self):
        return self.model_wrapper.u_dim

    def fetch_info_from_meter_dict(self, mode: str):
        info = {'iter_num': self.num_iter}
        for key, value in self.meter_dict[mode].items():
            round_number = 7 if 'loss' in key else 3
            info[key] = round(value.avg, round_number)
        return info

    def train(self, train_winding: bool, train_trajectory: bool):
        assert train_winding or train_trajectory
        logger.info('train mode: {}'.format('winding' if train_winding else 'trajectory'))
        self.initialize_meter_dict()
        end = time()
        self.model_wrapper.train_mode()

        for num_iter in tqdm(range(self.num_iter, self.max_iter)):
            if self.finished:
                break
            batch_dict = self.retrieve_data()
            for mode, batch in batch_dict.items():
                self.model_wrapper.send_batch_to_device(batch)
                ref_trajectory = batch['ref_trajectory']  # (batch_size, num_agents, num_ref_frames, dim_states)
                tar_trajectory = batch['tar_trajectory']  # (batch_size, num_agents, num_tar_frames, dim_states)
                tar_winding_onehot = batch['tar_winding_onehot']  # (batch_size, num_edges, dim_winding)
                dst_onehot = batch['dst_onehot'].squeeze()  # (batch_size, num_agents, dim_goal)
                tar_winding_index = batch['tar_winding_index']  # (batch_size, num_edges)
                dst_index = batch['dst_index']  # (batch_size, num_agents)
                self.meter_dict[mode]['data_time'].update(time() - end)

                total_loss = 0.
                if train_trajectory:
                    B, loss_dict, _ = self.model_wrapper.train_trajectory(
                        ref_trajectory, tar_trajectory, tar_winding_onehot, dst_onehot, self.losses)
                    self.optim_trajectory.zero_grad()
                    loss_dict['total_loss'].backward()
                    clip_grad_value_(self.trajectory_model.parameters(), self.config.gradient_clip)
                    self.optim_trajectory.step()

                    total_loss += loss_dict['total_loss']
                    self.meter_dict[mode]['node_loss'].update(loss_dict['node_loss'].item(), B)

                if train_winding:
                    B, loss_dict, _, _ = self.model_wrapper.train_winding(
                        ref_trajectory, tar_trajectory, tar_winding_index, dst_index, self.losses)
                    self.optim_winding.zero_grad()
                    loss_dict['total_loss'].backward()
                    clip_grad_value_(self.winding_model.parameters(), self.config.gradient_clip)
                    self.optim_winding.step()

                    total_loss += loss_dict['total_loss']
                    self.meter_dict[mode]['winding_loss'].update(loss_dict['winding_loss'].item(), B)
                    self.meter_dict[mode]['goal_loss'].update(loss_dict['goal_loss'].item(), B)

                self.meter_dict[mode]['total_loss'].update(total_loss.item(), B)
                self.meter_dict[mode]['batch_time'].update(time() - end)
                end = time()

                # Record information every x iterations
                if num_iter % self.config.iter_collect == 0:
                    info = self.fetch_info_from_meter_dict(mode)
                    self.tb_writer[mode].add_scalar('loss/total', info['total_loss'], num_iter)
                    self.tb_writer[mode].add_scalar('loss/node', info['node_loss'], num_iter)
                    self.tb_writer[mode].add_scalar('loss/edge', info['edge_loss'], num_iter)
                    self.tb_writer[mode].add_scalar('loss/winding', info['winding_loss'], num_iter)
                    self.tb_writer[mode].add_scalar('loss/goal', info['goal_loss'], num_iter)
                    self.tb_writer[mode].add_scalar('time/per_iter', info['batch_time'], num_iter)
                    self.tb_writer[mode].add_scalar('time/data_fetch', info['data_time'], num_iter)
                    for key in self.meter_dict[mode].keys():
                        self.meter_dict[mode][key] = AverageMeter()
                    end = time()

            if num_iter % self.config.save_every == 0:
                self.save(train_winding, train_trajectory)

            self.num_iter += 1

    def eval(self, data_loader) -> Dict[str, Any]:
        self.initialize_meter_dict()
        self.model_wrapper.eval_mode()

        output = []
        with torch.no_grad():
            t1 = time()
            for batch_index, batch in tqdm(enumerate(data_loader)):
                self.model_wrapper.send_batch_to_device(batch)

                ref_trajectory = batch[
                    'ref_trajectory']  # torch.float32, (batch_size, num_agents, num_ref_frames, dim_states)
                tar_trajectory = batch[
                    'tar_trajectory']  # torch.float32, (batch_size, num_agents, num_tar_frames, dim_states)
                tar_winding_index = batch['tar_winding_index']  # torch.int64, (batch_size, num_edges)
                src_index = batch['src_index']  # torch.int64, (batch_size, num_agents)
                dst_index = batch['dst_index']  # torch.int64, (batch_size, num_agents)
                batch_size, num_agents, num_rollout, dim_states = tar_trajectory.shape
                dim_goal = 4
                dim_winding = 2

                tar = torch.cat((dst_index, tar_winding_index), dim=-1)
                freq_list, sample_list = self.model_wrapper.eval_winding(ref_trajectory, src_index)

                curr_inputs = []
                pred_windings = []
                pred_goals = []
                num_items = []
                for item_index, (frequencies, unique_samples) in enumerate(zip(freq_list, sample_list)):
                    curr_input, pred_winding, pred_goal = prep_data(ref_trajectory, item_index, unique_samples,
                                                                    dim_goal)
                    num_items.append(curr_input.shape[0])
                    curr_inputs.append(curr_input)
                    pred_windings.append(pred_winding)
                    pred_goals.append(pred_goal)
                curr_input = torch.cat(curr_inputs, dim=0)
                pred_winding = torch.cat(pred_windings, dim=0)
                pred_goal = torch.cat(pred_goals, dim=0)
                pred_trajectory = self.model_wrapper.eval_trajectory(
                    curr_state=curr_input,
                    num_rollout=num_rollout,
                    winding_onehot=pred_winding,
                    goal_onehot=pred_goal)

                item_pos = 0
                for item_index, (frequencies, num_item) in enumerate(zip(freq_list, num_items)):
                    item = {
                        'src': ref_trajectory[item_index, :].squeeze(),  # (num_agents, num_ref_frames, dim_states)
                        'tar': {
                            'winding': tar[item_index, :],  # (num_agents * 2)
                            'trajectory': tar_trajectory[item_index, :].squeeze()
                        },
                        'prd': {
                            'frequency': frequencies,  # List[int], (num_unique_samples, )
                            'winding': pred_winding[item_pos:item_pos + num_item],
                            # torch.float32, (num_unique_samples, num_agents, dim_winding)
                            'goal': pred_goal[item_pos:item_pos + num_item],
                            # torch.float32, (num_unique_samples, num_agents, dim_goal)
                            'trajectory': pred_trajectory[item_pos:item_pos + num_item],
                            # torch.float32, (num_unique_samples, num_agents, num_tar_frames, dim_states)
                        }
                    }
                    output.append(item)
                    item_pos += num_item
                # if batch_index >= 4:
                #     break
            t2 = time()
            print('elapsed time {:03d}: {:7.5f}'.format(batch_index, t2 - t1))
        return output

    def save(self, train_winding, train_trajectory):
        save_dir = self.config.model_dir

        if train_winding:
            filename = save_dir / 'winding{}.pt'.format(self.num_iter)
            checkpoint = {
                'iter_num': self.num_iter,
                'epoch_num': self.num_epoch,
                'optim_winding': self.optim_winding.state_dict(),
                'winding_model': self.winding_model.state_dict(),
            }
            torch.save(checkpoint, str(filename))

        if train_trajectory:
            filename = save_dir / 'trajectory{}.pt'.format(self.num_iter)
            checkpoint = {
                'iter_num': self.num_iter,
                'epoch_num': self.num_epoch,
                'optim_trajectory': self.optim_trajectory.state_dict(),
                'trajectory_model': self.trajectory_model.state_dict(),
            }
            torch.save(checkpoint, str(filename))

    def load(self):
        winding_files = sorted(self.config.model_dir.glob('winding*.pt'),
                               key=lambda x: int(re.findall(r'([\d]+)', x.stem)[0]))
        trajectory_files = sorted(self.config.model_dir.glob('trajectory*.pt'),
                                  key=lambda x: int(re.findall(r'([\d]+)', x.stem)[0]))
        if winding_files:
            winding_file = winding_files[-1]
            checkpoint_winding = torch.load(str(winding_file), map_location='cpu')
            self.num_iter = max(self.num_iter, checkpoint_winding['iter_num'])
            self.num_epoch = max(self.num_epoch, checkpoint_winding['epoch_num'])
            self.optim_winding.load_state_dict(checkpoint_winding['optim_winding'])
            self.winding_model.load_state_dict(checkpoint_winding['winding_model'])

        if trajectory_files:
            trajectory_file = trajectory_files[-1]
            checkpoint_trajectory = torch.load(str(trajectory_file), map_location='cpu')
            self.num_iter = max(self.num_iter, checkpoint_trajectory['iter_num'])
            self.num_epoch = max(self.num_epoch, checkpoint_trajectory['epoch_num'])
            self.optim_trajectory.load_state_dict(checkpoint_trajectory['optim_trajectory'])
            self.trajectory_model.load_state_dict(checkpoint_trajectory['trajectory_model'])

        if winding_files or trajectory_files:
            logger.info('Loaded model and optimizer: epoch {}, iteration {}'.format(self.num_epoch, self.num_iter))
            logger.info('Configuration directory: {}'.format(self.config.model_dir))


def dst_combination(src_indices: List[int]):
    dst_candidates = [[j for j in range(4) if j != index] for index in src_indices]
    return list(product(*dst_candidates))
