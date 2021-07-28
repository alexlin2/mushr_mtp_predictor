import random
from argparse import Namespace, ArgumentParser
from pathlib import Path

import numpy as np
import torch

from mtp.config import ModelType, fetch_model_name
from mtp.utils.directory import mkdir_if_not_exists, fetch_model_dir, fetch_log_dir
from mtp.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_arguments(argparser):
    args = argparser.parse_args()

    args.device = 'cuda:{}'.format(args.gpu_index) if args.gpu_index >= 0 else 'cpu'

    if args.model_type == 'GraphNet':
        args.model_type = ModelType.GraphNet
    elif args.model_type == 'GraphNetFullyConnected':
        args.model_type = ModelType.GraphNetFullyConnected
    elif args.model_type == 'GRU':
        args.model_type = ModelType.GRU
    else:
        raise KeyError('invalid model_type: {}'.format(args.model_type))
    args.model_name = '-'.join([fetch_model_name(args.model_type)])
    exp_names = [
        'm-{}'.format(args.model_name),
        'n-{}'.format(args.num_agents),
        's-{}'.format(args.seed),
        'b-{:4.2f}'.format(args.beta),
    ]
    args.exp_name = '_'.join(exp_names)
    args.image_dir = mkdir_if_not_exists(Path.cwd() / '.gnn/images/{}'.format(args.exp_name))
    args.eff_num_agents = args.custom_num_agents if args.custom_num_agents >= 0 else args.num_agents
    args.model_dir = mkdir_if_not_exists(fetch_model_dir() / args.exp_name)
    args.train_log_dir = mkdir_if_not_exists(fetch_log_dir() / args.exp_name / 'train')
    args.test_log_dir = mkdir_if_not_exists(fetch_log_dir() / args.exp_name / 'test')
    return args


def fetch_arguments() -> Namespace:
    parser = ArgumentParser(description='MTP Training')
    parser.add_argument('-v', '--verbose', action='store_true', help='print debug information')
    parser.add_argument('-s', '--scenarios', metavar='S', default=2000, type=str, help='scenarios to collect for')
    parser.add_argument('-n', '--num-agents', default=2, type=int, help='Number of agents in a scenario')
    parser.add_argument('--num-exp', default=20, type=int, help='Total number of experiments')
    parser.add_argument("-a", "--agent", type=str, choices=["gn", "autopilot"], help="select which agent to run", default="gn")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--use_winding', action='store_true', dest='use_winding')
    parser.add_argument('--train-winding', action='store_true')
    parser.add_argument('--train-trajectory', action='store_true')
    parser.add_argument('--bsize', type=int, default=100, help='batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loader workers')
    parser.add_argument('--num-history', type=int, default=15)
    parser.add_argument('--num-rollout', type=int, default=25)
    parser.add_argument('--pred-type', type=str, default='cond')
    parser.add_argument('--u-dim', type=int, default=4)
    parser.add_argument('--use-condition', action='store_true', default=False)
    parser.add_argument('--predict-condition', action='store_true', default=False)
    parser.add_argument('--model-type', type=str, default='GraphNet')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max-iter', type=int, default=40000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--test-every', type=int, default=50)
    parser.add_argument('--pred-w-dim', type=int, default=2)
    parser.add_argument('--pred-g-dim', type=int, default=4)
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--custom-index', type=int, default=-1)
    parser.add_argument('--custom-num-agents', type=int, default=-1)

    return parse_arguments(parser)
