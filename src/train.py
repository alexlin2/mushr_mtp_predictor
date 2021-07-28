from pathlib import Path
from typing import List, Optional

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
    args = fetch_arguments()
    split_names = ['train', 'test']

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


def compute_min_ade(
        t1: Tensor,
        t2: Tensor,
        reduction: str = 'min') -> float:
    """
    Computes average displacement error
    :param t1: torch.float32, (num_unique_samples, num_agents, num_frames, dim_trajectory)
    :param t2: torch.float32, (num_unique_samples, num_agents, num_frames, dim_trajectory)
    :param reduction: str, specifies how to reduce the output ('min': minimum of each agent, 'scene': scene-level)
    :return: float
    """
    assert t1.ndim == 4 and t2.ndim == 4
    diff = t1[..., :2] - t2[..., :2]  # (num_unique_samples, num_agents, num_frames, 2)
    diff = torch.sqrt(torch.sum(torch.pow(diff, 2), -1))  # (num_unique_samples, num_agents, num_frames)
    diff = torch.mean(diff, dim=-1)  # (num_unique_samples, num_agents), average distance per sample and per agent
    if reduction == 'min':  # mean of minimum ADE of each agent
        min_value, _ = torch.min(diff, dim=0)
        return torch.mean(min_value).item()
    elif reduction == 'scene':  # scene-level ADE: minimum among average distance per sample
        mean_value = torch.mean(diff, dim=1)
        min_value = torch.min(mean_value)
        return min_value.item()
    else:
        raise KeyError('invalid reduction: {}'.format(reduction))


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


def draw_trajectories(
        src_trajectory: Tensor,
        dst_trajectory: Tensor,
        prd_trajectory: Tensor,
        frequency: List[int],
        image_dir: Optional[Path] = None,
        item_index: int = -1):
    """
    Draw trajectories
    :param src_trajectory: torch.float32, (num_agents, num_ref_states, dim_states)
    :param dst_trajectory: torch.float32, (num_agents, num_tar_states, dim_states)
    :param prd_trajectory: torch.float32, (num_unique_samples, num_agents, num_tar_states, dim_states)
    :param frequency: List[int], (num_unique_samples, )
    :param image_dir: Path, output path
    :param item_index: int
    :return:
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    src_array = src_trajectory.detach().cpu().numpy()
    dst_array = dst_trajectory.detach().cpu().numpy()
    prd_array = prd_trajectory.detach().cpu().numpy()
    num_agents, num_ref_states, dim_states = src_trajectory.shape
    num_unique_samples = prd_array.shape[0]
    for n in range(num_agents):
        plt.plot(src_array[n, :, 0], src_array[n, :, 1])
    for n in range(num_agents):
        plt.plot(dst_array[n, :, 0], dst_array[n, :, 1])

    text_offset = 0.1
    for s in range(num_unique_samples):
        for n in range(num_agents):
            plt.plot(prd_array[s, n, :, 0], prd_array[s, n, :, 1], alpha=0.3)
            plt.text(
                x=prd_array[s, n, -1, 0] + text_offset,
                y=prd_array[s, n, -1, 1] + text_offset,
                s='sample{}:{}'.format(s, frequency[s]),
                fontsize=8)

        x_center = 1.8
        y_center = 2.4
        margin = 1.6
        ax.set_xlim(left=x_center - margin, right=x_center + margin)
        ax.set_ylim(bottom=y_center - margin, top=y_center + margin)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(b=True)

        if image_dir is not None:
            image_path = image_dir / 'item{:05d}_sample{:02d}.png'.format(item_index, s)
            plt.savefig(str(image_path))
    plt.show()
    plt.close(fig)


def test():
    args = fetch_arguments()
    split_names = ['test']

    # args_pred.bsize = 1
    dc, lc, tc, model_dir = get_config_list(args)
    storage_dict = {name: fetch_trajectory_storage(num_agents=args.num_agents, is_train=name == 'train')
                    for name in split_names}
    dataset_dict = {name: TrajectoryDataset(storage=storage) for name, storage in storage_dict.items()}

    for split_name, dataset in dataset_dict.items():
        print(split_name, len(dataset))

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
    for item_index, output in enumerate(outputs):
        src_trajectory = output['src']
        tar_trajectory = output['tar']['trajectory'].to(gn_wrapper.device)  # torch.float32, (num_agents, num_tar_frames, dim_states)
        print('src', src_trajectory[:, -1, :])
        print('tar', tar_trajectory[:, -1, :])

        num_unique_samples = len(output['prd']['frequency'])
        pred_trajectory = output['prd']['trajectory']  # torch.float32, (num_unique_samples, num_agents, num_tar_frames, dim_states)
        pred_frequency = output['prd']['frequency']  # List[int], (num_unique_samples, )
        tiled_tar_trajectory = tar_trajectory.unsqueeze(0).expand(num_unique_samples, -1, -1, -1)
        print(pred_trajectory.shape)
        print(pred_frequency)
        print('prd', pred_trajectory[:5, :, -1, :])

        min_ade_min = compute_min_ade(pred_trajectory, tiled_tar_trajectory, reduction='min')
        min_ade_scene = compute_min_ade(pred_trajectory, tiled_tar_trajectory, reduction='scene')
        print('min_ade_min  : {:7.5f}'.format(min_ade_min))
        print('min_ade_scene: {:7.5f}'.format(min_ade_scene))

        draw_trajectories(src_trajectory, tar_trajectory, pred_trajectory, pred_frequency, args.image_dir, item_index)


if __name__ == '__main__':
    test()
    # train()
