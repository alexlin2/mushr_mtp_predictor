import threading

import torch

from collision_checker import CollisionChecker

class Tracking:

    def __init__(self, params, rollout_size = 15, check_collision = True):
        self.rollout_size = rollout_size
        self.collision_checker = CollisionChecker(params)
        self.collision_check = check_collision
        self.dtype = torch.FloatTensor

        self.NPOS = 4
        self.lookahead = 1.0
        self._prev_pose = None
        self._prev_index = -1
        self._cache_thresh = 0.01

        self.error_w = 100.0

        self.path_lock = threading.Lock()
        with self.path_lock:
            self.path = None

    def apply(self, world, preds, probs):
        _, waypoint = self._get_reference_index(world.ego_pose)
        probs = torch.Tensor(1. / probs)
        preds = torch.from_numpy(preds)
        ego_preds = preds[:, 0, :]
        errorcost = ego_preds[:, self.rollout_size - 1, :2].sub(torch.Tensor(waypoint[:2])).norm(dim=1).mul(self.error_w)
        min_cost_id = torch.argmin(errorcost)
        return min_cost_id, errorcost

    def set_task(self, pathmsg):
        """
        Args:
        path [(x,y,h,v),...] -- list of xyhv named tuple
        """
        self._prev_pose = None
        self._prev_index = None
        path = self.dtype([[pathmsg[i].x, pathmsg[i].y, pathmsg[i].h, pathmsg[i].v] for i in range(len(pathmsg))])
        assert path.size() == (len(pathmsg),4)

        with self.path_lock:
            self.path = path
            self.waypoint_diff = torch.mean(torch.norm(
                self.path[1:, :2] - self.path[:-1, :2], dim=1))

            return True

    def _get_reference_index(self, pose):
        '''
        get_reference_index finds the index i in the controller's path
            to compute the next control control against
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - referencence index
        '''
        with self.path_lock:
            if ((self._prev_pose is not None) and
                    (torch.norm(self._prev_pose[:2] - pose[:2]) < self._cache_thresh)):
                return (self._prev_index, self.path[self._prev_index])
            diff = self.path[:, :3] - pose
            dist = diff[:, :2].norm(dim=1)
            index = dist.argmin()
            index += int(self.lookahead / self.waypoint_diff)
            index = min(index, len(self.path)-1)
            self._prev_pose = pose
            self._prev_index = index
            return (index, self.path[index])