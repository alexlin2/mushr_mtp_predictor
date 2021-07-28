import math
from itertools import combinations, permutations
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mtp.data.trajectory_storage import TrajectoryStorage, fetch_trajectory_storage


def one_hot_vectors_from_indices(indices: List[int], num_labels: int) -> np.ndarray:
    """
    Returns one-hot encoded stacked vectors from indices
    :@param indices: List[int], number of indices
    :@param num_labels: int, number of total labels
    :@return: np.ndarray, np.float32, (num_indices, num_labels), one-hot encoded stacked vectors
    """
    value = np.zeros((len(indices), num_labels), dtype=np.float32)
    for d, i in enumerate(indices):
        value[d, i] = 1.
    return value


def one_hot_vectors_from_winding_numbers(winding_numbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes one-hot encoded winding number values
    :@param winding_numbers: np.float32, (num_edges, )
    :return:
        np.float32, (num_edges, 2), one-hot encoded winding number values
        np.int64, (num_edges, ), one-hot indices as ground-truth label
    """
    winding_indices = [int(v > 0) for v in winding_numbers]
    return one_hot_vectors_from_indices(winding_indices, 2), np.array(winding_indices, dtype=np.int64)


def compute_winding_numbers(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the winding number from a pair of trajectory values
    :@param a: np.ndarray, np.float32, (num_frames_a, 4)
    :@param b: np.ndarray, np.float32, (num_frames_b, 4)
    :@return: np.ndarray, np.float32, (min_num_frames, ), winding number values
    """
    # assume that the first two dimensions of a and b are (x, y) position values
    min_length = min(len(a), len(b))
    aa = a[:min_length, ...]
    bb = b[:min_length, ...]
    zx = aa[1:, 0] - bb[1:, 0]
    dzx = zx - (aa[:-1, 0] - bb[:-1, 0])
    zy = aa[1:, 1] - bb[1:, 1]
    dzy = zy - (aa[:-1, 1] - bb[:-1, 1])
    theta_new = (np.multiply(dzy, zx) - np.multiply(zy, dzx)) / (np.power(zx, 2) + np.power(zy, 2) + 1.0e-20)
    theta_new = theta_new / (2 * math.pi)
    return float(np.sum(theta_new))
    # integral = np.cumsum(theta_new)
    # return np.insert(integral, 0, 0.)


def compute_winding_numbers_vector(trajectory: np.ndarray) -> np.ndarray:
    """
    :@param trajectory: np.float32, (num_agents, num_frames, 4)
    :@return: np.float32, (num_edges, ), winding number values
    """
    permutation_indices = list(permutations(range(trajectory.shape[0]), 2))
    winding_values = np.zeros((len(permutation_indices), ))
    for i, (j, k) in enumerate(permutation_indices):
        winding_values[i] = compute_winding_numbers(trajectory[j, ...], trajectory[k, ...])
    return winding_values


class TrajectoryDataset(Dataset):
    def __init__(self, storage: TrajectoryStorage):
        self.storage = storage
        self.num_ref_frames = 15
        self.num_tar_frames = 25

    def __getitem__(self, index: int):
        trajectory = self.storage.get_trajectory(index)  # (num_agents, dim_trajectory, num_frames)
        trajectory = np.transpose(trajectory, (0, 2, 1))  # (num_agents, num_frames, dim_trajectory)
        intention = self.storage.get_intention(index)  # (num_agents, dim_intention)

        ref_trajectory = trajectory[:, :self.num_ref_frames, :]  # (num_agents, num_ref_frames, dim_trajectory)
        tar_trajectory = trajectory[:, self.num_ref_frames:, :]  # (num_agents, num_tar_frames, dim_trajectory)
        src_index = intention[:, 0]  # (num_agents, )
        dst_index = intention[:, 1]  # (num_agents, )
        dst_onehot = one_hot_vectors_from_indices(dst_index, 4)  # (num_agents, 4)
        winding_values = compute_winding_numbers_vector(ref_trajectory)  # (num_edges, )
        winding_onehot, winding_indices = one_hot_vectors_from_winding_numbers(winding_values)  # (num_edges, 2)

        item = {
            'ref_trajectory': ref_trajectory,
            'tar_trajectory': tar_trajectory,
            'tar_winding': winding_values,
            'tar_winding_onehot': winding_onehot,
            'tar_winding_index': winding_indices,
            'src_index': src_index,
            'dst_index': dst_index,
            'dst_onehot': dst_onehot,
        }
        return {k: torch.tensor(v) for k, v in item.items()}

    def __len__(self):
        return len(self.storage)


if __name__ == '__main__':
    storage = fetch_trajectory_storage(num_agents=2, is_train=True)
    dataset = TrajectoryDataset(storage)
    keys = {'src_index', 'dst_index', 'tar_winding'}
    for i in range(100):
        item = dataset[i]
        indices = np.concatenate((item['src_index'], item['dst_index']), axis=-1)
        winding = int(item['tar_winding'][0] > 0)
        print(indices, winding)
