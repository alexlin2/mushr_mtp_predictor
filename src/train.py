#!/usr/bin/env python3
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from mtp.argument import fetch_arguments
from mtp.config import get_config_list
from mtp.data.dataset import TrajectoryDataset
from mtp.data.trajectory_storage import fetch_trajectory_storage
from mtp.networks import fetch_model_iterator
from mtp.train import Trainer


def train():
    args = fetch_arguments(is_train=True)
    split_names = ['train', 'test']

    # args_pred.bsize = 1
    dc, lc, tc, model_dir = get_config_list(args)
    storage_dict = {name: fetch_trajectory_storage(num_agents=args.num_agents, is_train=name == 'train')
                    for name in split_names}
    dataset_dict = {name: TrajectoryDataset(storage=storage) for name, storage in storage_dict.items()}
    data_loader_dict = {name: DataLoader(
        dataset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True) for name, dataset in dataset_dict.items()}

    run_every = {'test': 1, 'train': 100}
    gn_wrapper = fetch_model_iterator(lc, args)
    trainer = Trainer(gn_wrapper, split_names, data_loader_dict, run_every, tc)
    trainer.train(train_winding=args.train_winding, train_trajectory=args.train_trajectory)


def compute_ade(t1: Tensor, t2: Tensor) -> float:
    """
    Computes average displacement error
    :param t1: torch.float32, (num_agents, num_frames, dim_trajectory)
    :param t2: torch.float32, (num_agents, num_frames, dim_trajectory)
    :return: float
    """
    diff = t1[..., :2] - t2[..., :2]  # (num_agents, num_frames, 2)
    diff = torch.sqrt(torch.sum(torch.pow(diff, 2), -1))  # (num_agents, num_frames)
    return torch.mean(diff).item()


def compute_fde(t1: Tensor, t2: Tensor) -> float:
    """
    Computes final displacement error
    :param t1: torch.float32, (num_agents, num_frames, dim_trajectory)
    :param t2: torch.float32, (num_agents, num_frames, dim_trajectory)
    :return: float
    """
    diff = t1[..., -1, :2] - t2[..., -1, :2]  # (num_agents, 2)
    diff = torch.sqrt(torch.sum(torch.pow(diff, 2), -1))  # (num_agents, )
    return torch.mean(diff).item()


def test():
    args = fetch_arguments(is_train=False)
    split_names = ['test']

    # args_pred.bsize = 1
    dc, lc, tc, model_dir = get_config_list(args)
    storage_dict = {name: fetch_trajectory_storage(num_agents=args.num_agent, is_train=name == 'train')
                    for name in split_names}
    dataset_dict = {name: TrajectoryDataset(storage=storage) for name, storage in storage_dict.items()}
    data_loader_dict = {name: DataLoader(
        dataset,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True) for name, dataset in dataset_dict.items()}

    run_every = {'test': 1}
    gn_wrapper = fetch_model_iterator(lc, args)
    trainer = Trainer(gn_wrapper, split_names, data_loader_dict, run_every, tc)
    trainer.load()
    outputs = trainer.eval(data_loader_dict['test'])
    for i, output in enumerate(outputs):
        tar_trajectory = output['tar']['trajectory'].to(gn_wrapper.device)
        print(i, type(output))
        print('tar:trajectory', output['tar']['trajectory'].shape)
        ade_list = []
        fde_list = []
        for j, p in enumerate(output['prd']):
            prd_trajectory = p['trajectory'].squeeze()
            ade = compute_ade(tar_trajectory, prd_trajectory)
            fde = compute_fde(tar_trajectory, prd_trajectory)
            ade_list.append(ade)
            fde_list.append(fde)
        min_ade = torch.min(torch.tensor(ade_list))
        min_fde = torch.min(torch.tensor(fde_list))
        print('min_ade', min_ade)
        print('min_fde', min_fde)

        for k, v in output.items():
            if isinstance(v, Tensor):
                print(k, v.shape, v.dtype)
            else:
                print(k, type(v))


if __name__ == '__main__':
    test()
    # train()
