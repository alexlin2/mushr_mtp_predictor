import json
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import List, Dict, Any, Callable

import numpy as np
import torch

from mtp.utils.directory import fetch_agent_data_path_list, fetch_agent_split_path, fetch_agent_intermediate_data_dir, \
    fetch_train_intermediate_trajectory_path, fetch_test_intermediate_trajectory_path


def read_single_raw_data(data_path: Path) -> List[Dict[str, Any]]:
    """
    Read data from the single .pt file
    returns a list of dictionary of data with keys
    {'intention_{}', 'x{}_data', 'y{}_data', 'yaw{}_data', 'speed{}_data', 'heuristic{}_data', ...}
        * 'intention': Tuple[int, int], specifies the initial and the goal positions as indices
        * 'x': (n, ), x positions
        * 'y': (n, ), y positions
        * 'yaw': (n, ), yaw values
        * 'speed': (n, ), speed values
        * 'heuristic': (n, ), bool, ???
    """
    return list(torch.load(str(data_path)))


def read_raw_data_files(num_agents: int) -> List[Dict[str, Any]]:
    """
    Read raw data files w.r.t the number of agents
    """
    data_files: List[Path] = fetch_agent_data_path_list(num_agents)
    return list(chain.from_iterable(map(read_single_raw_data, data_files)))


def split_raw_data_index_dict(
        num_agents: int,
        num_items: int,
        train_ratio: float = 0.8) -> Dict[str, List[int]]:
    split_filepath = fetch_agent_split_path(num_agents)
    if split_filepath.exists():
        with open(str(split_filepath), 'r') as file:
            split_index_dict = json.load(file)
    else:
        num_train_items = int(train_ratio * num_items)
        indices = np.random.permutation(list(range(num_items))).tolist()
        train_indices = indices[:num_train_items]
        test_indices = indices[num_train_items:]
        split_index_dict = {
            'train': train_indices,
            'test': test_indices,
        }
        with open(str(split_filepath), 'w') as file:
            json.dump(split_index_dict, file, indent=4)
    return split_index_dict


def reshape_array_with_fixed_frames(
        data: np.ndarray,
        num_frames: int):
    if len(data) < num_frames:
        return None
    num_items = len(data) - num_frames + 1
    indices = np.expand_dims(np.arange(num_frames), 0) + np.expand_dims(np.arange(num_items), 0).T
    return data[indices]


def transform_raw_data_dict_into_array(
        raw_data_list: List[Dict[str, Any]],
        num_agents: int,
        window_size: int,
        path_func: Callable[[int, int], Path]):
    data_dict = defaultdict(list)
    intention_dict = defaultdict(list)
    data_key_fmt = ['x{}_data', 'y{}_data', 'yaw{}_data', 'speed{}_data']
    data_key_lists = [[f.format(i) for f in data_key_fmt] for i in range(num_agents)]
    intention_keys = ['intention_{}'.format(i) for i in range(num_agents)]
    file_index = 0
    for raw_data in raw_data_list:
        for agent_index, (data_keys, intention_key) in enumerate(zip(data_key_lists, intention_keys)):
            stacked_trajectory = np.stack([reshape_array_with_fixed_frames(raw_data[data_key], window_size)
                                           for data_key in data_keys])  # (#keys, #trajectories, wsize)
            data_dict[agent_index].append(stacked_trajectory)
            intention_dict[agent_index].append([raw_data[intention_key] for _ in range(len(stacked_trajectory))])
        if len(data_dict[agent_index]) >= 10000:
            data_path = path_func(num_agents, file_index)
            data = {
                'trajectory': data_dict,
                'intention': intention_dict,
            }
            torch.save(data, str(data_path))
            data_dict = defaultdict(list)
            intention_dict = defaultdict(list)
            file_index += 1


def save_intermediate_trajectory_array_dict(
        num_agents: int,
        num_ref_frames: int = 15,
        num_tar_frames: int = 25,
        train_ratio: float = 0.8):
    data_dir = fetch_agent_intermediate_data_dir(num_agents)
    train_path_list = sorted(data_dir.glob('train*.pt'))
    if train_path_list:
        return

    raw_data: List[Dict[str, Any]] = read_raw_data_files(num_agents)
    split_index_dict: Dict[str, List[int]] = split_raw_data_index_dict(
        num_agents=num_agents,
        num_items=len(raw_data),
        train_ratio=train_ratio)
    raw_data_dict: Dict[str, List[Dict[str, Any]]] = \
        {k: [raw_data[i] for i in v] for k, v in split_index_dict.items()}
    train_raw_data = raw_data_dict['train']
    test_raw_data = raw_data_dict['test']

    window_size = num_ref_frames + num_tar_frames
    transform_raw_data_dict_into_array(train_raw_data, num_agents, window_size, fetch_train_intermediate_trajectory_path)
    transform_raw_data_dict_into_array(test_raw_data, num_agents, window_size, fetch_test_intermediate_trajectory_path)
