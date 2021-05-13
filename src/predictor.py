import numpy as np
import pandas as pd
import torch

class GraphNetPredictor:
    def __init__(self, trainer, u_dim, B, N, T, rollout_size, d):
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
        fig = plt.figure()
        # ax = plt.gca()
        ax = p3.Axes3D(fig)
        ax.view_init(90, -90)
        ax.set_xlim((200, 300))
        ax.set_ylim((-200, -300))
        # # ax.set_zlim((0, 20))
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])
        # plt.pause(1.)
        self.eval_root_dir = Path.cwd() / '.gnn/eval'

    def set_src_dst_tensor(self, global_intent):
        self.global_intent = global_intent
        src_index_tensor = [3]
        for i in range(2, len(global_intent), 2):
            src_index_tensor.append(int(global_intent[i]))
        src_index_tensor = torch.tensor(
            [src_index_tensor]).to(self.trainer.device)
        self.src_index_tensor = src_index_tensor
        # dst_index = []
        # for i in range(1, len(global_intent), 2):
        #     dst_index.append(int(global_intent[i]))
        # self.dst_index = torch.tensor(
        #     [dst_index]).to(self.trainer.device)
        # self.dst_list = self.trainer.generate_dst_combinations(src_index_tensor)

        # self.winding = torch.zeros([self.B, self.N*(self.N-1), 2]).to(self.trainer.device)
        # self.dest = torch.zeros([self.B, self.N, 4]).to(self.trainer.device)

    def get_input(self, world):
        curr_state = np.empty((self.B, self.N, self.T, self.d))
        for i, player in enumerate(world.agents):
            past_hist_np = np.expand_dims(np.asarray(player.past_hist), axis=0)
            curr_state[0][i] = past_hist_np
        # plt.plot(curr_state[0, 0, :, 0], -1 * curr_state[0, 0, :, 1])
        curr_state = torch.from_numpy(curr_state).float().to(self.trainer.device)
        return curr_state

    def stop(self):
        plt.savefig("rollouts.png")
        plt.clf()
        if len(self.inputs) > 0:
            output = torch.stack(self.inputs)
            torch.save(output, "./inputs.pt")
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        # ax = plt.gca()
        ax.view_init(90, -90)
        ax.set_xlim((200, 300))
        ax.set_ylim((-200, -300))
        # ax.set_zlim((0, 20))
        self.prev_pred = None

    def predict(self, world, step, dt, scene_id, behavior_id, param_id):
        threshold = 0.0
        output_prd = []
        probs = []
        output = []
        with torch.no_grad():
            curr_state = self.get_input(world)  # Shape: [B x n x T x d]
            # print (curr_state.size())
            self.inputs.append(curr_state)
            next_state = torch.zeros((self.B, self.N, self.rn, self.d)).to(
                self.trainer.device)  # Shape: [B x n x rollout_num x d]
            tar_winding = torch.zeros((self.B, self.N * (self.N - 1))).to(
                self.trainer.device)  # B, E
            src_index_tensor = self.src_index_tensor.to(
                self.trainer.device)
            tar_goal = torch.zeros((self.B, self.N)).to(
                self.trainer.device)  # B, n
            winding_onehot = torch.zeros((self.B, self.N * (self.N - 1), 2)).to(
                self.trainer.device)  # B, E, 2
            goal_onehot = torch.zeros((self.B, self.N, 4)).to(
                self.trainer.device)  # B, n, 4

            num_goal = tar_goal.shape[-1]
            num_winding = tar_winding.shape[-1]
            tar = torch.cat((tar_goal, tar_winding), dim=-1)
            prd_cond = self.trainer.model_wrapper.eval_winding(curr_state, tar_winding.shape[0], tar_goal.shape[0])

            for r, rows in enumerate(prd_cond):
                item = {
                    'src': curr_state[r, :].squeeze(),
                    'tar': {'winding': tar[r, :], 'trajectory': next_state[r, :].squeeze()},
                    'prd': []
                }
                # for prd_goal_onehot, prd_winding_onehot, row in zip(dests, winds, rows):
                for freq, row in rows:
                    prd_goal = row[:num_goal]
                    prd_goal[0] = 1
                    prd_winding = row[num_goal:]
                    # skip if the goal position label is same with the starting position label
                    valid = True
                    if int(prd_goal[0]) != int(self.global_intent[1]):
                        valid = False
                    # # if int(prd_goal[1]) != int(self.global_intent[3]):
                    # #     continue
                    for g, s in zip(prd_goal, src_index_tensor[r, :]):
                        if g.item() == s.item():
                            valid = False

                    # skip if the winding numbers are inconsistent
                    for i1, i2 in self.winding_constraints:
                        if prd_winding[i1] != prd_winding[i2]:
                            valid = False
                    if not valid:
                        continue

                    prd_goal_onehot = onehot_from_index(prd_goal, 4).unsqueeze(0)
                    prd_winding_onehot = onehot_from_index(prd_winding, 2).unsqueeze(0)
                    # print ("wind: ", prd_winding_onehot)
                    # print ("dest: ", prd_goal_onehot)
                    curr_input = curr_state[r, ...].squeeze().unsqueeze(0)
                    B, prd_traj = self.trainer.model_wrapper.eval_trajectory(
                        curr_input, next_state.shape[2], prd_winding_onehot, prd_goal_onehot)
                    if float(freq) / 100 > threshold:
                        output_prd.append(prd_traj[0].cpu().numpy())
                        probs.append(float(freq) / 100)
                    item['prd'].append({
                        'frequency': freq,
                        'winding': row,
                        'trajectory': prd_traj,
                    })
                    # print ("TRAJ SIZE: ", prd_traj.size())
                output.append(item)

            eval_dir = self.eval_root_dir / f'scenario_{scene_id}' / f'behavior_{behavior_id}_{param_id}'
            mkdir_if_not_exists(eval_dir)
            if len(output) > 0:
                for item_id, item in enumerate(output):
                    visualize_single_row(eval_dir, step, item_id, item)
            # output_prd = output_prd
            output_prd = np.array(output_prd)
            # print("PROBS: ", probs)
            probs = np.array(probs)
            current_time = step * dt
            t_arr = np.arange(25) * 0.1 + current_time
            if len(probs) > 0:
                for i in range(output_prd.shape[0]):
                    for k in range(self.N):
                        output_prd[i, k, :, 0] += 257.5
                        output_prd[i, k, :, 1] += 247.5
                        output_prd[i, k, :, 1] = -1 * output_prd[i, k, :, 1]
                        output_prd[i, k, :, 3] = -1 * output_prd[i, k, :, 3]
                        output_prd[i, k, :self.rn - 1, 2] = output_prd[i, k, 1:, 0] - output_prd[i, k, :self.rn - 1, 0]
                        output_prd[i, k, :self.rn - 1, 3] = output_prd[i, k, 1:, 1] - output_prd[i, k, :self.rn - 1, 1]
                        dx, dy = output_prd[i, k, :, 2], output_prd[i, k, :, 3]
                        output_prd[i, k, :, 3] = np.sqrt(output_prd[i, k, :, 2] ** 2 + output_prd[i, k, :, 3] ** 2) / (
                            0.1)
                        output_prd[i, k, :, 2] = (-(np.arctan2(dy, dx) * 180 / np.pi) + 360.) % 360.
                self.prev_pred = output_prd
                self.prev_probs = probs
                return output_prd, probs
            else:
                return self.prev_pred, self.prev_probs