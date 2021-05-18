#!/usr/bin/env python3
import numpy as np
import torch
import rospy
from custom_classes import World, GraphNetPredictor

car_names = ["car30", "car38"]



if __name__ == "__main__":

    rospy.init_node("test")
    world = World(car_names, 2, 10, 4)
    world.setup_pub_sub()
    rate = rospy.Rate(10)

    

    while not rospy.is_shutdown():

        rate.sleep()
    