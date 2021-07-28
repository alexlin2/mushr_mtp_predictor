import json
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from argparse import ArgumentParser
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict

import train
from argument import Argument, modify_user_arguments
from config import get_config_list, fetch_model_type
from data.data_loader import get_trajectory_data_loader
from networks import fetch_model_iterator
from utils.directory import mkdir_if_not_exists
from utils.logging import get_logger

# from mtp.utils.tb_log_reader import generate_loss_figure
# from visualize import visualize_array_data

logger = get_logger(__name__)


def train_model(args: Argument):
    args.mode = 'train'
    dc, lc, tc, _ = get_config_list(args)
    gn_wrapper = fetch_model_iterator(lc, args)
    modes = ['train', 'test']
    dataloader = {m: get_trajectory_data_loader(
        dc,
        test=m == 'train',
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True) for m in modes}
    run_every = {'train': 1, 'test': args.test_every}
    trainer = train.Trainer(gn_wrapper, modes, dataloader, run_every, tc)

    train_winding = False
    train_trajectory = True

    trainer.train(train_winding, train_trajectory)
    trainer.save(train_winding, train_trajectory)


def eval_models(args: Argument):
    args.mode = 'test'
    dc, lc, tc, model_dir = get_config_list(args)

    modes = ['test']
    dataloader = {'test': get_trajectory_data_loader(
        dc,
        test=True,
        batch_size=args.bsize,
        num_workers=args.num_workers,
        shuffle=True)}
    run_every = {'test': 1}
    gn_wrapper = fetch_model_iterator(lc, args)
    trainer = train.Trainer(gn_wrapper, modes, dataloader, run_every, tc)
    output = trainer.eval(dataloader['test'])
    return trainer.num_iter, output


def collect_all_json_files(root_dir: Path):
    dirs = root_dir.glob('*')
    perf_dict = dict()
    for in_dir in dirs:
        if not in_dir.is_dir():
            continue
        exp_name = in_dir.stem
        pt_files = list(filter(lambda x: x.suffix in ['.pt', '.pth'], in_dir.glob('*')))
        json_files = list(in_dir.glob('*.json'))

        iter_list = [re.findall(r'iter([\d]+)', str(x)) for x in pt_files]
        if not iter_list:
            continue
        last_iter = sorted([int(x[0]) for x in iter_list])[-1]

        for json_file in json_files:
            name = '{}/{}'.format(in_dir.stem, json_file.stem)
            with open(str(json_file), 'r') as file:
                data = {int(k): v for k, v in json.load(file).items()}
            last_key = sorted(data.keys())[-1]
            if last_key != last_iter:
                print('not all the checkpoints were evaluated: {}, {}, {}'.format(name, last_key, last_iter))
            items = sorted([(k, v['total_loss']) for k, v in data.items()], key=itemgetter(0))
            perf_dict[name] = items
            items = sorted([(k, v['total_loss']) for k, v in data.items()], key=itemgetter(1))
            best_iter, best_loss = items[0]
            print(exp_name, json_file.stem, '{}/{}: {:6.4f}'.format(best_iter, last_key, best_loss))
    return perf_dict


def train_arg_modifier(args: Argument, user_args):
    args.CUDA_VISIBLE_DEVICES = user_args.gpu_index
    args.OMP_NUM_THREADS = 1
    args.num_agents = user_args.num_agents
    args.seed = user_args.seed
    args.model_type = user_args.model_type
    args.beta = user_args.beta
    if user_args.num_agents == 4:
        args.bsize = 20
        args.num_history = 5
        args.num_rollout = 15
    return args


def test_arg_modifier(args: Argument, user_args):
    args.CUDA_VISIBLE_DEVICES = -1
    args.OMP_NUM_THREADS = 1
    args.num_agents = user_args.num_agents
    args.seed = user_args.seed
    args.model_type = user_args.model_type
    args.beta = user_args.beta
    if user_args.num_agents == 4:
        args.bsize = 20
        args.num_history = 5
        args.num_rollout = 15
    return args


def train_agent(user_args):
    args = modify_user_arguments(partial(train_arg_modifier, user_args=user_args))
    logger.info('train {}'.format(args.exp_name))
    train_model(args)


def visualize_single_row(out_dir: Path, index: int, item_dict: Dict[str, Any]):
    out_path = out_dir / 'row{:03d}.png'.format(index)
    src_trajectory = item_dict['src'].squeeze().cpu().detach().numpy()
    tar_winding = item_dict['tar']['winding'].cpu().detach().numpy()
    tar_trajectory = item_dict['tar']['trajectory'].cpu().detach().numpy()

    num_plot = len(item_dict['prd']) + 1
    fig, axl = plt.subplots(nrows=1, ncols=num_plot, figsize=(num_plot * 3, 3))
    num_agents = tar_trajectory.shape[0]
    tar_trajectory = np.concatenate((src_trajectory, tar_trajectory), axis=1)
    for n in range(num_agents):
        axl[0].scatter(tar_trajectory[n, :, 0], tar_trajectory[n, :, 1])

    for j, (item, ax) in enumerate(zip(item_dict['prd'], axl[1:])):
        frequency = item['frequency']
        prd_winding = item['winding'].squeeze().cpu().detach().numpy()
        prd_trajectory = item['trajectory'].squeeze().cpu().detach().numpy()
        prd_trajectory = np.concatenate((src_trajectory, prd_trajectory), axis=1)
        for n in range(num_agents):
            ax.scatter(prd_trajectory[n, :, 0], prd_trajectory[n, :, 1])
            ax.set_title(str(frequency) + ', ' + ''.join([str(v) for v in prd_winding]))
            ax.grid()
            ax.axis('equal')

    # http://omz-software.com/pythonista/matplotlib/users/customizing.html
    font_family = 'Lato'
    font_weight = 'medium'
    rc_params = {
        'font.family': font_family,
        'font.weight': font_weight,
        'axes.labelweight': font_weight,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'axes.titlepad': 10,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'lines.linewidth': 2,
    }
    sns.set(style='whitegrid', rc=rc_params)


    plt.savefig(str(Path.home() / 'Downloads/test.png'))


def eval_agent(user_args):
    args = modify_user_arguments(partial(test_arg_modifier, user_args=user_args))
    args.bsize = 10
    logger.info(args.exp_name)

    num_iter, output = eval_models(args)
    eval_root_dir = Path.cwd() / '.gnn/eval'
    eval_dir = mkdir_if_not_exists(eval_root_dir / args.exp_name / 'iter{:05d}'.format(num_iter))
    for i, item in enumerate(output):
        visualize_single_row(eval_dir, i, item)
        break


def switch_modes():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str)
    parser.add_argument('--model_str', type=str, default='')
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.5)
    args = parser.parse_args()

    # if args.mode == 'vis':
    #     generate_loss_figure()
    if args.mode in ['train', 'eval']:
        args.model_type = fetch_model_type(args.model_str)
        print(args.mode, args.model_type, args.num_agents, args.seed, args.beta)
        func = train_agent if args.mode == 'train' else eval_agent
        func(args)
    else:
        raise TypeError('invalid mode: {}'.format(args.mode))


if __name__ == '__main__':
    switch_modes()
