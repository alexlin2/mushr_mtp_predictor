from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import (
    PoseStamped,
)
from scipy.spatial.transform import Rotation as R

import rospy
import numpy as np
import math

WB = 0.295
k = 0.6

class state:

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class PurePursuitController:

    def __init__(self, car_name, Lfc = 0.2) -> None:
        self.car_pose = state()
        self.Lc = Lfc
        self.Lfc = Lfc
        self.pub_ctrls = rospy.Publisher(
            "/"
            + car_name
            + "/"
            + "mux/ackermann_cmd_mux/input/navigation"
            ,
            AckermannDriveStamped,
            queue_size=2,
        )
        rospy.Subscriber(
            "/" + car_name + "/" + "car_pose",
            PoseStamped,
            self.cb_pose,
            queue_size=10,
        )

    
    def search_target_index(self, v, traj) -> int:
        self.Lf = self.Lfc + v * k
        idx = 0
        while self.Lf > self.car_pose.calc_distance(traj[idx,0], traj[idx,1]):
            if (idx + 1) >= len(traj):
                break
            idx += 1
        return idx

    def pure_pursuit_steer_control(self, traj) -> tuple:
        v = traj[4,3]
        idx = self.search_target_index(v, traj)
        tar_x, tar_y = traj[idx, :2]
        alpha = np.arctan2(tar_y - self.car_pose.rear_y, tar_x - self.car_pose.rear_x) - self.car_pose.yaw
        delta = np.arctan2(2.0 * WB * np.sin(alpha) / self.Lf, 1.0)
        print(v, delta)
        return v, delta

    def cb_pose(self, msg) -> None:
        q = msg.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        yaw = r.as_euler('zyx')[0]
        self.car_pose.update(msg.pose.position.x, msg.pose.position.y, yaw)

    def pub_command(self, crtl) -> None:
        v, delta = crtl

        drive = AckermannDrive(steering_angle=delta, speed=v)
        self.pub_ctrls.publish(AckermannDriveStamped(drive=drive))




