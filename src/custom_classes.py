#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import rospy 
from time import time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    PoseStamped,
    Quaternion,
)
from collections import deque
from itertools import product, permutations
from pathlib import Path

from scipy.spatial.transform import Rotation as R
from mtp.train import Trainer


def onehot_from_index(index_vector: torch.Tensor, num_cls: int):
    onehot = torch.zeros((index_vector.numel(), num_cls), dtype=torch.float32, device=index_vector.device)
    for i in range(index_vector.numel()):
        onehot[i, index_vector[i]] = 1.
    return onehot


def fetch_winding_constraints(num_agent: int):
    edge_index = list(permutations(range(num_agent), 2))
    sorted_edge_index = np.array([sorted(p) for p in edge_index])
    unique_edge_index = np.unique(sorted_edge_index, axis=0)
    constraints = [tuple(int(v) for v in np.where((sorted_edge_index == unique_edge_index[r]).all(axis=1))[0])
                   for r in range(unique_edge_index.shape[0])]
    return constraints

def prep_data(ref_trajectory, item_index: int, unique_samples, dim_goal: int):
    """
    Prepare the data for trajectory prediction from the samples
    :param ref_trajectory: torch.float32, (batch_size, num_agents, num_ref_frames, dim_states)
    :param item_index: index in the batch
    :param unique_samples: torch.float32, (num_unique_samples, num_agents * (dim_goal + dim_winding))
    :param dim_goal: int, dimension of one-hot goal vector
    :return:
        :curr_input: torch.float32, (num_unique_samples, num_agents, num_ref_frames, dim_states)
        :pred_winding: torch.float32, (num_unique_samples, num_agents, dim_winding)
        :pred_goal: torch.float32, (num_unique_samples, num_agents, dim_goal)
    """
    num_unique_samples = unique_samples.shape[0]
    num_agents = unique_samples.shape[1]
    unique_samples = unique_samples.view(num_unique_samples, num_agents, -1)
    pred_goal = unique_samples[..., :dim_goal]  # (num_unique_samples, num_agents, dim_goal)
    pred_winding = unique_samples[..., dim_goal:]  # (num_unique_samples, num_agents, dim_winding)
    curr_input = ref_trajectory[item_index, ...].squeeze().unsqueeze(0). \
        expand(num_unique_samples, -1, -1,
            -1)  # (num_unique_samples, num_agents, num_ref_frames, dim_states)
    return curr_input, pred_winding, pred_goal


class World:
    def __init__(self, car_names, N, T, d):
        self.T = T
        self.d = d
        self.N = N
        self.dtype = torch.FloatTensor
        self.car_names = car_names
        self.past_hist = np.zeros((N,T,d))
        self.past_poses = []  
        self.ego_pose = self.dtype([0,0,0])
        self.populated = False
        assert T > 5 
        assert len(car_names) == N

        for i in range(N):
            self.past_poses.append(deque(maxlen=5))

    def setup_pub_sub(self):
        for car in self.car_names:
            rospy.Subscriber(
                "/" + car + "/" + "car_pose",
                PoseStamped,
                self.callback_populate_hist,
                callback_args=car,
                queue_size=10,
            )

    def callback_populate_hist(self, msg, car_name):
        player_index = self.car_names.index(car_name)
        self.past_poses[player_index].appendleft(msg)
        prev_poses = self.past_poses[player_index]

        if np.count_nonzero(self.past_hist[player_index]) == self.T * self.d:
            self.populated = True

        if player_index == 0:
            q = msg.pose.orientation
            r = R.from_quat([q.x, q.y, q.z, q.w])
            yaw = r.as_euler('zyx')[0]
            self.ego_pose = self.dtype([msg.pose.position.x, msg.pose.position.y, yaw])

        if len(prev_poses) == 5:
            dx = prev_poses[0].pose.position.x - prev_poses[-1].pose.position.x
            dy = prev_poses[0].pose.position.y - prev_poses[-1].pose.position.y
            dt = (prev_poses[0].header.stamp - prev_poses[-1].header.stamp).to_sec()
        else:
            dx = 0.0
            dy = 0.0
            dt = 1.0

        xydxdy = msg.pose.position.x, msg.pose.position.y, dx/dt, dy/dt

        prev_hist = np.copy(self.past_hist[player_index])
        prev_hist[1:] = prev_hist[:-1]
        prev_hist[0] = np.array(xydxdy)
        self.past_hist[player_index] = prev_hist

    def get_past_hist(self):
        return self.past_hist


class GraphNetPredictor:
    def __init__(self, trainer: Trainer, u_dim, B, N, T, rollout_size, d):
        self.trainer = trainer
        self.u_dim = u_dim
        self.B = 1
        self.N = N
        self.T = T
        self.rn = rollout_size
        self.d = d
        self.winding_constraints = fetch_winding_constraints(N)
        self.inputs = []
        self.prev_pred = None
        self.prev_probs = None
        # fig = plt.figure()
        # # ax = plt.gca()
        # ax = p3.Axes3D(fig)
        # ax.view_init(90, -90)
        # ax.set_xlim((200, 300))
        # ax.set_ylim((-200, -300))
        # # ax.set_zlim((0, 20))
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        # plt.pause(1.)
        self.eval_root_dir = Path.cwd() / '.gnn/eval'

    def get_input(self, world):
        curr_state = np.empty((self.B, self.N, self.T, self.d))
        curr_state[0] = world.past_hist
        # plt.plot(curr_state[0, 0, :, 0], -1 * curr_state[0, 0, :, 1])
        curr_state = torch.from_numpy(curr_state).float().to(self.trainer.device)
        return curr_state



    def predict(self, world):
        output = []
        with torch.no_grad():

            dim_goal = 4
            # ref_trajectory = batch['ref_trajectory']  # Shape: [B x n x T x d]
            # tar_trajectory = batch['tar_trajectory']  # Shape: [B x n x rollout_num x d]
            # tar_winding_onehot = batch['tar_winding_onehot']  # B, E, 2
            # dst_onehot = batch['dst_onehot'].squeeze()  # B, n, 4
            # tar_winding_index = batch['tar_winding_index']  # B, E
            # dst_index = batch['dst_index']  # B, n

            curr_state = self.get_input(world)  # Shape: [B x n x T x d]

            #print(curr_state)

            # #self.inputs.append(curr_state)
            # next_state = torch.zeros((self.B, self.N, self.rn, self.d)).to(
            #     self.trainer.device)  # Shape: [B x n x rollout_num x d]
            # tar_winding = torch.zeros((self.B, self.N * (self.N - 1))).to(
            #     self.trainer.device)  # B, E
            # # src_index_tensor = self.src_index_tensor.to(
            # #     self.trainer.device)
            # tar_goal = torch.zeros((self.B, self.N)).to(
            #     self.trainer.device)  # B, n
            # winding_onehot = torch.zeros((self.B, self.N * (self.N - 1), 2)).to(
            #     self.trainer.device)  # B, E, 2
            # goal_onehot = torch.zeros((self.B, self.N, 4)).to(
            #     self.trainer.device)  # B, n, 4

            # num_goal = tar_goal.shape[-1]
            # num_winding = tar_winding.shape[-1]
            # tar = torch.cat((tar_goal, tar_winding), dim=-1)

            freq_list, sample_list = self.trainer.model_wrapper.eval_winding(curr_state)

            # for r, rows in enumerate(prd_cond):
            for item_index, (frequencies, unique_samples) in enumerate(zip(freq_list, sample_list)):
                item = {
                    'src': curr_state[item_index, :].squeeze(),
                    'prd': []
                }
        
                curr_input, pred_winding, pred_goal = prep_data(curr_state, item_index, unique_samples,
                                                                    dim_goal)
        
                
                # pred_goal = torch.tensor([[[1.,0.,0.,0.],[0.,0.,1.,0.]]]).to(self.trainer.device)
                # pred_winding = torch.tensor([[1.0,0.0],[0.0,1.0]]).to(self.trainer.device)
                # pred_goal = pred_goal.expand(len(unique_samples), -1, -1).to(self.trainer.device)
                # pred_goal_onehot = onehot_from_index(pred_goal, 4).unsqueeze(0)
                # pred_winding_onehot = onehot_from_index(pred_winding, 2).unsqueeze(0)
    
                pred_trajectory = self.trainer.model_wrapper.eval_trajectory(
                        curr_state=curr_input,
                        num_rollout=self.rn,
                        winding_onehot=pred_winding,
                        goal_onehot=pred_goal)  # torch.float32, (num_unique_samples, num_agents, num_tar_frames, dim_states)
                item['prd'] = {
                    'frequency': frequencies,  # List[int], (num_unique_samples, )
                    'winding': pred_winding,  # torch.float32, (num_unique_samples, num_agents, dim_winding)
                    'goal': pred_goal,  # torch.float32, (num_unique_samples, num_agents, dim_goal)
                    'trajectory': pred_trajectory,  # torch.float32, (num_unique_samples, num_agents, num_tar_frames, dim_states)
                }
                output.append(item)

            output_prd = np.array([x['prd']['trajectory'].cpu().numpy() for x in output])[0]

            probs = np.array([x['prd']['frequency'] for x in output])[0]
            goals = np.array([x['prd']['goal'] for x in output])[0]
            #print(output_prd[0,0])
    
            if len(probs) > 0:
                for i in range(output_prd.shape[0]):
                    for k in range(self.N):
                        output_prd[i, k, :self.rn - 1, 2] = output_prd[i, k, 1:, 0] - output_prd[i, k, :self.rn - 1, 0]
                        output_prd[i, k, :self.rn - 1, 3] = output_prd[i, k, 1:, 1] - output_prd[i, k, :self.rn - 1, 1]
                        dx, dy = output_prd[i, k, :, 2], output_prd[i, k, :, 3]
                        output_prd[i, k, :, 3] = np.sqrt(output_prd[i, k, :, 2] ** 2 + output_prd[i, k, :, 3] ** 2) / (
                            0.1)
                        output_prd[i, k, :, 2] = np.arctan2(dy, dx)
                self.prev_pred = output_prd
                self.prev_probs = probs
                return output_prd, probs, goals
            else:
                return self.prev_pred, self.prev_probs, goals
