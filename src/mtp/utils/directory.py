from pathlib import Path
from typing import Tuple, List

def mkdir_if_not_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True)
    if not path.exists():
        raise FileNotFoundError('the directory was not found: {}'.format(path))
    return path


def fetch_data_root_dir() -> Path:
    data_root_dir = Path.home() / 'data/mtp'
    assert data_root_dir.exists()
    return data_root_dir


def fetch_agent_raw_data_dir(num_agents: int) -> Path:
    assert num_agents in {2, 3, 4}
    data_dir = fetch_data_root_dir() / '{}agent_100'.format(num_agents)
    assert data_dir.exists()
    return data_dir


def fetch_agent_intermediate_data_dir(num_agents: int) -> Path:
    data_dir = fetch_data_root_dir() / '{}agent_inter'.format(num_agents)
    return mkdir_if_not_exists(data_dir)


def fetch_trajectory_storage_dir(num_agents: int) -> Path:
    return mkdir_if_not_exists(fetch_data_root_dir() / '{}agent'.format(num_agents))


def fetch_train_trajectory_storage_path(num_agents: int) -> Path:
    return mkdir_if_not_exists(fetch_trajectory_storage_dir(num_agents) / 'train')


def fetch_test_trajectory_storage_path(num_agents: int) -> Path:
    return mkdir_if_not_exists(fetch_trajectory_storage_dir(num_agents) / 'test')


def fetch_agent_data_path_list(num_agents: int) -> List[Path]:
    data_dir = fetch_agent_raw_data_dir(num_agents)
    return sorted(data_dir.glob('*.pt'))


def fetch_agent_split_path(num_agents: int) -> Path:
    return fetch_agent_raw_data_dir(num_agents) / 'split.json'


def fetch_train_intermediate_trajectory_path(num_agents: int, index: int) -> Path:
    return fetch_agent_intermediate_data_dir(num_agents) / 'train{:03d}.pt'.format(index)


def fetch_test_intermediate_trajectory_path(num_agents: int, index: int) -> Path:
    return fetch_agent_intermediate_data_dir(num_agents) / 'test{:03d}.pt'.format(index)


def fetch_root_dir() -> Path:
    working_directory = '/home/alexlin/catkin_ws/src/mushr_mtp_predictor/src/'
    return mkdir_if_not_exists(Path(working_directory) / '.gnn')


def fetch_data_dir() -> Path:
    return mkdir_if_not_exists(fetch_root_dir() / 'data')


def fetch_model_dir() -> Path:
    return mkdir_if_not_exists(fetch_root_dir() / 'weights')


def fetch_log_dir() -> Path:
    return mkdir_if_not_exists(fetch_root_dir() / 'logs')


def fetch_raw_traj_path(num_agent: int) -> Path:
    if num_agent in {2, 3, 4}:
        return fetch_data_dir() / 'raw_traj_data_{}agents.csv'
    else:
        raise ValueError('the number of agent should be in [2, 3, 4]: {}'.format(num_agent))


def fetch_traj_path(num_agent: int) -> Path:
    if num_agent in {2, 3, 4}:
        return fetch_data_dir() / 'traj_data{}.csv'
    else:
        raise ValueError('the number of agent should be in [2, 3, 4]: {}'.format(num_agent))


def fetch_traj_path_pair(self, num_agent: int) -> Tuple[Path, Path]:
    return fetch_raw_traj_path(num_agent), self.fetch_traj_path(num_agent)


def fetch_index_path(num_agent: int, mode: str) -> Path:
    return fetch_data_dir() / 'index_{}{}.json'.format(mode, num_agent)


def fetch_data_path(num_agent: int, mode: str) -> Path:
    return fetch_data_dir() / 'data_{}{}.pt'.format(mode, num_agent)
