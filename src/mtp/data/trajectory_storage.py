from math import sqrt
from pathlib import Path
from typing import Callable, Any, Tuple

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from mtp.utils.directory import fetch_train_trajectory_storage_path, fetch_agent_intermediate_data_dir, \
    fetch_test_trajectory_storage_path


def key_str(hash: int) -> str:
    return '{:12d}'.format(hash)


def encode_str(query: str) -> bytes:
    return query.encode()


def decode_str(byte_str: bytes) -> str:
    return byte_str.decode('utf-8')


def encode_array(query: np.ndarray) -> bytes:
    return query.tobytes()


def decode_array(byte_str: bytes, dtype, shape) -> np.ndarray:
    return np.frombuffer(byte_str, dtype=dtype).reshape(*shape)


def decode_trajectory(byte_str: bytes, num_agents: int) -> np.ndarray:
    return decode_array(byte_str, np.float32, (num_agents, 4, -1))


def decode_intention(byte_str: bytes) -> np.ndarray:
    return decode_array(byte_str, np.int64, (-1, 2))


class TrajectoryStorage:
    def __init__(
            self,
            num_agents: int,
            read_only: bool = True,
            db_path: Path = None):
        self.num_agents = num_agents
        self.db_path = db_path
        self.num_dbs = 2
        self.env = lmdb.open(
            path=str(self.db_path),
            max_dbs=self.num_dbs,
            map_size=5e11,
            max_readers=1,
            readonly=read_only,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.trajectories = self.env.open_db(encode_str('trajectories'))
        self.intentions = self.env.open_db(encode_str('intentions'))

    def _get_num_items(self, db) -> int:
        with self.env.begin(db=db, write=False) as txn:
            return txn.stat()['entries']

    def __len__(self):
        return self._get_num_items(self.trajectories)

    def key(self, index: int):
        return encode_str(key_str(index))

    def _get_item(
            self,
            db,
            index: int,
            decode_func: Callable[[Any], Any]):
        with self.env.begin(db=db, write=False) as txn:
            return decode_func(txn.get(self.key(index)))

    def _put_item(
            self,
            db,
            index: int,
            item,
            encode_func):
        with self.env.begin(db=db, write=True) as txn:
            txn.put(self.key(index), encode_func(item))

    def decode_trajectory(self, byte_str: bytes) -> np.ndarray:
        return decode_trajectory(byte_str, self.num_agents)

    def get_trajectory(self, index: int) -> np.ndarray:
        return self._get_item(
            db=self.trajectories,
            index=index,
            decode_func=self.decode_trajectory)

    def put_trajectory(self, index: int, trajectory: np.ndarray):
        return self._put_item(
            db=self.trajectories,
            index=index,
            item=trajectory,
            encode_func=encode_array)

    def get_intention(self, index: int) -> np.ndarray:
        return self._get_item(
            db=self.intentions,
            index=index,
            decode_func=decode_intention)

    def put_intention(self, index: int, intention: np.ndarray):
        self._put_item(
            db=self.intentions,
            index=index,
            item=intention,
            encode_func=encode_array)


def normalize_angle(raw_angle: float) -> float:
    angle = (raw_angle + np.pi) % np.pi
    if angle > 0.5 * np.pi:
        angle -= np.pi
    return angle


def check_local_validity(
        local_trajectories: np.ndarray,
        num_ref_frames: int) -> bool:
    """
    Check if the local trajectories are valid
    :param local_trajectories: np.float32, (num_agents, dim_states, window_size)
    :param num_ref_frames: int
    :return: bool
    """
    len_threshold = 0.5
    angle_threshold = np.pi / 180 * 3

    num_agents = local_trajectories.shape[0]
    local_valid_list = []
    for n in range(num_agents):
        # check if the reference trajectory is within the intersection area
        # we set (1.4 x 1.4) area at the center of (1.8, 2.4)
        ref_trajectory = local_trajectories[n, :2, :num_ref_frames]  # np.float32, (2, num_ref_frames)
        tar_trajectory = local_trajectories[n, :2, num_ref_frames:]  # np.float32, (2, num_tar_frames)
        x = ref_trajectory[0, :]
        y = ref_trajectory[1, :]
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        if (min_x < 1.1 and max_x > 2.5) or (min_y < 1.7 and max_y > 3.1):
            local_valid_list.append(False)
            continue

        # check if the length of reference or target trajectory is too short
        # if any case is found, completely ignore the sample
        rfdx = ref_trajectory[0, -1] - ref_trajectory[0, 0]
        rfdy = ref_trajectory[1, -1] - ref_trajectory[1, 0]
        tfdx = tar_trajectory[0, -1] - tar_trajectory[0, 0]
        tfdy = tar_trajectory[1, -1] - tar_trajectory[1, 0]
        rfl = sqrt(rfdx * rfdx + rfdy * rfdy)
        tfl = sqrt(tfdx * tfdx + tfdy * tfdy)
        if rfl < len_threshold or tfl < len_threshold:
            return False

        # check if the trajectories are too flat
        dx = ref_trajectory[0, 1:] - ref_trajectory[0, :-1]
        dy = ref_trajectory[1, 1:] - ref_trajectory[1, :-1]
        angles = np.arctan2(dy, dx)
        min_angle = normalize_angle(np.min(angles))
        max_angle = normalize_angle(np.max(angles))
        mid_angle = normalize_angle(np.median(angles))
        if np.abs(max_angle - mid_angle) < angle_threshold or np.abs(mid_angle - min_angle) < angle_threshold:
            local_valid_list.append(False)
            continue
        else:
            local_valid_list.append(True)
    return np.any(np.array(local_valid_list))


def check_validity(
        trajectories: np.ndarray,
        intentions: np.ndarray,
        num_ref_frames: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check the validity of the trajectory arrays and returns pruned trajectory
    :param trajectories: np.float32, (num_items, num_agents, dim_states, window_size)
    :param intentions: np.int64, (num_items, num_agents, dim_intentions)
    :param num_ref_frames: int,
    :return:
        valid_trajectories: np.float32, (num_valid_items, num_agents, dim_states, window_size)
        valid_intentions: np.int64, (num_valid_items, num_agents, dim_intentions)
    """
    num_items = trajectories.shape[0]
    valid_indices = [i for i in range(num_items) if check_local_validity(trajectories[i, ...], num_ref_frames)]
    return trajectories[valid_indices], intentions[valid_indices]


def create_storage_from_intermediate_data(
        num_agents: int,
        num_ref_frames: int = 15):
    path_func_dict = {
        'train': fetch_train_trajectory_storage_path,
        'test': fetch_test_trajectory_storage_path,
    }
    path_query_dict = {
        'train': 'train*.pt',
        'test': 'test*.pt',
    }

    for split_name in {'train', 'test'}:
        db_path = path_func_dict[split_name](num_agents)
        query = path_query_dict[split_name]
        storage = TrajectoryStorage(
            num_agents=num_agents,
            read_only=False,
            db_path=db_path)
        storage_index = 0
        train_inter_paths = sorted(fetch_agent_intermediate_data_dir(num_agents).glob(query))
        for p in train_inter_paths:
            data_dict = torch.load(str(p))
            trajectory_dict = data_dict['trajectory']
            intention_dict = data_dict['intention']
            num_items = len(trajectory_dict[0])
            for row in tqdm(range(num_items)):
                trajectories = []
                intentions = []
                for agent_index in range(num_agents):
                    trajectory: np.ndarray = trajectory_dict[agent_index][row]  # (4, #items, window_size)
                    trajectory = np.transpose(trajectory, (1, 0, 2))  # (#items, 4, window_size)
                    intention: Tuple[int, int] = intention_dict[agent_index][row][0]
                    intention: np.ndarray = np.tile(np.array(intention)[..., None], trajectory.shape[0]).T  # (#items, 2)
                    trajectories.append(trajectory)
                    intentions.append(intention)
                trajectories = np.stack(trajectories, axis=1).astype(np.float32)  # (#items, num_agents, 4, window_size)
                intentions = np.stack(intentions, axis=1).astype(np.int64)  # (#items, num_agents, 2)

                # refine trajectories and intentions
                refined_trajectories, refined_intentions = check_validity(trajectories, intentions, num_ref_frames)
                for local_index in range(refined_trajectories.shape[0]):
                    storage.put_trajectory(storage_index, refined_trajectories[local_index])
                    storage.put_intention(storage_index, refined_intentions[local_index])
                    storage_index += 1


def fetch_trajectory_storage(
        num_agents: int,
        is_train: bool) -> TrajectoryStorage:
    path_func = fetch_train_trajectory_storage_path if is_train else fetch_test_trajectory_storage_path
    return TrajectoryStorage(num_agents=num_agents, read_only=True, db_path=path_func(num_agents))


if __name__ == '__main__':
    create_storage_from_intermediate_data(2)
