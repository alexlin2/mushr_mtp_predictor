#!/usr/bin/env python3
import numpy as np
import torch
import rospy
from custom_classes import World, GraphNetPredictor
from torch import Tensor
from torch.utils.data import DataLoader

from mtp.argument import fetch_arguments
from mtp.config import get_config_list
from mtp.data.dataset import TrajectoryDataset
from mtp.data.trajectory_storage import fetch_trajectory_storage
from mtp.networks import fetch_model_iterator
from mtp.train import Trainer

car_names = ["car30", "car38"]
T = 15
B = 1
num_agents = 2

def get_trainer():
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
    return trainer


if __name__ == "__main__":

    rospy.init_node("test")
    world = World(car_names, num_agents, T, 4)
    world.setup_pub_sub()
    rate = rospy.Rate(10)
    trainer = get_trainer()
    predictor = GraphNetPredictor(trainer, u_dim=4, B=1, N=num_agents, T=T, rollout_size=25,
                                  d=4)

    while not rospy.is_shutdown():
        output_prd, probs = predictor.predict(world)
        print(output_prd)
        rate.sleep()
    