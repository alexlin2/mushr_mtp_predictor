import matplotlib.cm as cm
import matplotlib.colors as colors
from numpy import cosh
import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, Float32

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

_traj_pub = rospy.Publisher(
    "debug/viz_rollouts",
    MarkerArray,
    queue_size=1,
)

_waypoint_pub = rospy.Publisher(
    "debug/viz_waypoint/topic",
    PoseStamped,
    queue_size=10,
)

_path_viz_pub = rospy.Publisher(
        "debug/viz_path", PoseArray, queue_size=10)

def angle_to_rosquaternion(theta):
    r = R.from_euler('z', theta)
    x, y, z, w = r.as_quat()[0], r.as_quat()[1], r.as_quat()[2], r.as_quat()[3]
    return Quaternion(x=x, y=y, z=z, w=w)

def viz_trajs_cmap(poses, costs, ns="trajs", cmap="gray", scale=0.03):
    max_c = torch.max(costs)
    min_c = torch.min(costs)

    norm = colors.Normalize(vmin=min_c, vmax=max_c)
    # if cmap not in cm.cmaps_listed.keys():
    #     cmap = "viridis"
    cmap = cm.get_cmap(name=cmap)

    def colorfn(cost):
        r, g, b, a = 0.0, 0.0, 0.0, 1.0
        col = cmap(norm(cost))
        r, g, b = col[0], col[1], col[2]
        if len(col) > 3:
            a = col[3]
        return r, g, b, a

    return viz_trajs(poses, costs, colorfn, ns, scale)


def viz_trajs(poses, costs, colorfn, ns="trajs", scale=0.03):
    """
        poses should be an array of trajectories to plot in rviz
        costs should have the same dimensionality as poses.size()[0]
        colorfn maps a point to an rgb tuple of colors
    """
    assert poses.shape[0] == costs.shape[0]

    markers = MarkerArray()

    for i, (traj, cost) in enumerate(zip(poses, costs)):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = ns + str(i)
        m.id = i
        m.type = m.LINE_STRIP
        m.action = m.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = scale
        m.color.r, m.color.g, m.color.b, m.color.a = colorfn(cost)

        for t in traj:
            p = Point()
            p.x, p.y = t[0], t[1]
            m.points.append(p)

        markers.markers.append(m)

    _traj_pub.publish(markers)

def viz_path(path):
    poses = []
    for i in range(0, len(path)):
        p = Pose()
        p.position.x = path[i].x
        p.position.y = path[i].y
        p.orientation = angle_to_rosquaternion(path[i].h)
        poses.append(p)
    pa = PoseArray()
    pa.header = Header()
    pa.header.stamp = rospy.Time.now()
    pa.header.frame_id = "map"
    pa.poses = poses
    _path_viz_pub.publish(pa)

def viz_selected_waypoint(pose):
    p = PoseStamped()
    p.header = Header() 
    p.header.stamp = rospy.Time.now()
    p.header.frame_id = "map"
    p.pose.position.x = pose[0]
    p.pose.position.y = pose[1]
    heading = pose[2] % np.pi
    heading = (heading + np.pi) % np.pi
    p.pose.orientation = angle_to_rosquaternion(pose[2])
    _waypoint_pub.publish(p)