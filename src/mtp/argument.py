import random
from argparse import Namespace, ArgumentParser
from functools import partial
from os import environ

import numpy as np
import torch

from mtp.config import ModelType, fetch_model_name
from mtp.utils.logging import get_logger

logger = get_logger(__name__)


CUDA_INDEX = 0
NUM_THREAD = 5
NUM_AGENT = 3
U_DIM = 4
W_DIM = 1
NUM_HISTORY = 15
NUM_ROLLOUT = 25
PRED_TYPE = 'cond'
USE_CONDITION = False
PREDICT_CONDITION = False
LR = 1e-3
BETA = 0.5
GRADIENT_CLIP = 1.0
BSIZE = 200
NUM_WORKERS = 6
SEED = 0
MAX_ITER = 40000
SAVE_EVERY = 1000
TEST_EVERY = 50


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Argument:
    def __init__(self):
        self.CUDA_VISIBLE_DEVICES = CUDA_INDEX
        self.OMP_NUM_THREADS = NUM_THREAD
        self.num_agent = NUM_AGENT
        self.num_history = NUM_HISTORY
        self.num_rollout = NUM_ROLLOUT
        self.pred_type = PRED_TYPE
        self.u_dim = U_DIM
        self.use_condition = USE_CONDITION
        self.predict_condition = PREDICT_CONDITION
        self.model_type = ModelType.GraphNet
        self.bsize = BSIZE
        self.num_workers = NUM_WORKERS
        self.starts_from = -1
        self.custom_index = -1
        self.custom_num_agent = -1
        self.seed = SEED
        self.lr = LR
        self.beta = BETA
        self.gradient_clip = GRADIENT_CLIP
        self.mode = 'train'
        self.max_iter = MAX_ITER
        self.save_every = SAVE_EVERY
        self.test_every = TEST_EVERY
        self.pred_w_dim = 2
        self.pred_g_dim = 4

    @property
    def device(self) -> str:
        return 'cuda:{}'.format(self.CUDA_VISIBLE_DEVICES) if self.CUDA_VISIBLE_DEVICES >= 0 else 'cpu'

    @property
    def model_name(self) -> str:
        names = [fetch_model_name(self.model_type)]
        # if not self.use_condition:
        #     names += ['uncond']
        # if self.predict_condition:
        #     names += ['pred']
        return '-'.join(names)

    @property
    def exp_name(self):
        exp_names = [
            'm-{}'.format(self.model_name),
            # 't-{}'.format(self.pred_type),
            'n-{}'.format(self.num_agent),
            's-{}'.format(self.seed),
            'b-{:4.2f}'.format(self.beta),
        ]
        return '_'.join(exp_names)


_default_args = Argument()


def fetch_user_arguments() -> Namespace:
    argparser = ArgumentParser(description='Training MTP')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '-s', '--scenarios',
        metavar='S',
        default=2000,
        type=str,
        help='scenarios to collect for')
    argparser.add_argument(
        '-n', '--num-agent',
        default=2,
        type=int,
        help='Number of agents in a scenario')
    argparser.add_argument(
        '--num-exp',
        default=20,
        type=int,
        help='Total number of experiments')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["gn", "autopilot"],
                           help="select which agent to run",
                           default="gn")
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--beta', type=float, default=0.5)
    argparser.add_argument('--use_winding', action='store_true', dest='use_winding')
    argparser.add_argument('--train-winding', action='store_true')
    argparser.add_argument('--train-trajectory', action='store_true')
    argparser.add_argument('--bsize', type=int, default=100, help='batch size')
    argparser.add_argument('--num-workers', type=int, default=4, help='number of data loader workers')
    argparser.add_argument('--gpu_index', type=int, default=0)
    args = argparser.parse_args()
    return args


def modify_user_arguments(modifier=None) -> Argument:
    args = _default_args

    if modifier is not None:
        args = modifier(args)

    if isinstance(args.CUDA_VISIBLE_DEVICES, int) or isinstance(args.CUDA_VISIBLE_DEVICES, str):
        environ['CUDA_VISIBLE_DEVICES'] = str(args.CUDA_VISIBLE_DEVICES)
    elif isinstance(args.CUDA_VISIBLE_DEVICES, list):
        environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(v) for v in args.CUDA_VISIBLE_DEVICES])
    else:
        logger.error('invalid type of CUDA_VISIBLE_DEVICES: {}'.format(type(args.CUDA_VISIBLE_DEVICES)))
        raise TypeError()

    environ['OMP_NUM_THREADS'] = str(args.OMP_NUM_THREADS)
    return args


def test_arg_modifier(args, user_args):
    args.CUDA_VISIBLE_DEVICES = 0
    args.OMP_NUM_THREADS = 1
    args.num_agent = user_args.num_agent
    args.seed = user_args.seed
    args.bsize = user_args.bsize
    # args.model_type = user_args.model_type
    args.beta = user_args.beta
    if user_args.num_agent == 4:
        args.bsize = 20
        args.num_history = 5
        args.num_rollout = 15
    return args


def fetch_arguments(is_train: bool) -> Argument:
    user_args = fetch_user_arguments()
    modifier = None if is_train else partial(test_arg_modifier, user_args=user_args)
    args = modify_user_arguments(modifier)
    return args
