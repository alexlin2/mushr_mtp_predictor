#!/usr/bin/env python3
import numpy as np
import torch
import rospy
from custom_classes import World, GraphNetPredictor
from torch import Tensor
from torch.utils.data import DataLoader
from cost import Tracking
from parameters import RosParams

from mushr_rhc_ros.srv import FollowPath
from mushr_rhc_ros.msg import XYHVPath, XYHV
from utils2.viz_traj import viz_trajs_cmap

from mtp.argument import fetch_arguments
from mtp.config import get_config_list
from mtp.data.dataset import TrajectoryDataset
from mtp.data.trajectory_storage import fetch_trajectory_storage
from mtp.networks import fetch_model_iterator
from mtp.train import Trainer

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header, Float32
import rospy
import numpy as np


car_names = ["car30", "car38"]
T = 15
B = 1
num_agents = 2
rollout_size = 25

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

class PredictionNode():
    
    def __init__(self, trainer, world, predictor, cost_function, params):
        self.trainer = trainer
        self.world = world
        self.predictor = predictor
        self.cost_function = cost_function
        self.params = params
        self.task_set = False

        rospy.Service("~task/path", FollowPath, self.srv_path)

    def run(self):
        if self.task_set:
            preds, probs = self.predictor.predict(self.world)
            idx, error = self.cost_function.apply(self.world, preds, probs)
            
            _, idx = torch.sort(error)
            viz_trajs_cmap(preds[:,0,:][idx], error[idx], ns="result")


    def srv_path(self, msg):
        path = msg.path.waypoints
        self.cost_function.set_task(path)
        self.task_set = True
        print("Path Set")
        return True

if __name__ == "__main__":

    rospy.init_node("test")

    rate = rospy.Rate(10)
    params = RosParams()

    world = World(car_names, num_agents, T, 4)
    world.setup_pub_sub()
    
    trainer = get_trainer()
    predictor = GraphNetPredictor(trainer, u_dim=4, B=1, N=num_agents, T=T, rollout_size=rollout_size,
                                  d=4)
    cost_function = Tracking(params, rollout_size=rollout_size)
    
    node = PredictionNode(trainer, world, predictor, cost_function, params)


    np.set_printoptions(precision=2)
    while not rospy.is_shutdown():
        node.run()
        rate.sleep()
    