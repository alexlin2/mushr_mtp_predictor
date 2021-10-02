#!/usr/bin/env python3
import numpy as np
import torch
import rospy
from custom_classes import World, GraphNetPredictor
from controller import PurePursuitController
from torch.utils.data import DataLoader
from cost import Tracking
from parameters import RosParams

from mushr_rhc_ros.srv import FollowPath
from mushr_rhc_ros.msg import XYHVPath, XYHV
from utils2.viz_traj import viz_trajs_cmap, viz_path

from mtp.argument import fetch_arguments
from mtp.config import get_config_list
from mtp.data.dataset import TrajectoryDataset
from mtp.data.trajectory_storage import fetch_trajectory_storage
from mtp.networks import fetch_model_iterator
from mtp.train import Trainer
import matplotlib.pyplot as plt

import rospy
import numpy as np


car_names = ["car30", "car38"]
T = 15
B = 1
num_agents = 2
rollout_size = 25

def draw_trajectories(prd_trajectory, frequency):


    prd_array = prd_trajectory
    num_agents = prd_array.shape[1]
    num_unique_samples = prd_array.shape[0]

    plt.ion()
    text_offset = 0.1
    for s in range(num_unique_samples):
        for n in range(num_agents):
            plt.plot(prd_array[s, n, :, 0], prd_array[s, n, :, 1], alpha=0.3)

    plt.pause(0.5)
    plt.show()

def get_trainer() -> Trainer:
    args = fetch_arguments()
    split_names = ['test']

    # args_pred.bsize = 1
    lc, tc = get_config_list(args)
    storage_dict = {name: fetch_trajectory_storage(num_agents=args.num_agents, is_train=name == 'train')
                    for name in split_names}
    dataset_dict = {name: TrajectoryDataset(storage=storage)
                    for name, storage in storage_dict.items()}

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
    return trainer



class PredictionNode:    
    def __init__(self, trainer, world, predictor, cost_function, params) -> None:
        self.trainer = trainer
        self.world = world
        self.predictor = predictor
        self.cost_function = cost_function
        self.params = params
        self.task_set = False
        self.target_waypoint = None
        self.outputs = []
        self.traj = None

        rospy.Service("~task/path", FollowPath, self.srv_path)

    def run(self):
        if self.task_set and self.world.populated:
            preds, probs, goal = self.predictor.predict(self.world)
            print(preds[:,0].shape)
            idx, error = self.cost_function.apply(self.world, preds, probs)
            self.outputs.append((preds, probs))
            #print(len(preds))
            self.traj = preds[idx, 0]
            
            
            viz_trajs_cmap(preds[:,0,:], error, ns="result")
            if self.check_task_complete():
                return False
        return True

    def srv_path(self, msg):
        path = msg.path.waypoints
        self.cost_function.set_task(path)
        self.task_set = True
        self.target_waypoint = np.array([path[-1].x,path[-1].y]) 
        viz_path(path)
        print("Path Set, target waypoint: " + str(self.target_waypoint))
        return True

    def check_task_complete(self):
        pose = self.world.ego_pose
        target = self.target_waypoint
        dist = np.linalg.norm(pose[:2]-target)
        return dist < 0.5

if __name__ == "__main__":

    rospy.init_node("test")

    rate = rospy.Rate(10)
    params = RosParams()

    world = World(car_names, num_agents, T, 4)
    world.setup_pub_sub()
    controller = PurePursuitController("car30")
    
    trainer = get_trainer()
    predictor = GraphNetPredictor(trainer, u_dim=4, B=1, N=num_agents, T=T, rollout_size=rollout_size,
                                  d=4)
    cost_function = Tracking(params, rollout_size=rollout_size)
    
    node = PredictionNode(trainer, world, predictor, cost_function, params)

    state_dict = {}
    idx = 0

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    while not rospy.is_shutdown():
        # if idx < 100 and node.task_set:
        #     idx += 1
        #     data = world.get_past_hist()
        #     print(data)
        #     state_dict[str(idx)] = data.copy()
        # elif idx == 100:
        #     print(state_dict)
        #     torch.save(state_dict, str(Path.cwd()) + "/agent_trajectory.pt")  
        #     rospy.signal_shutdown("done")
        if not node.run():
            # for preds, probs in node.outputs:
            #     draw_trajectories(preds, probs)
            rospy.signal_shutdown("done")
        elif node.traj is not None:
            cmd = controller.pure_pursuit_steer_control(node.traj)
            controller.pub_command(cmd)
        rate.sleep()
    
    