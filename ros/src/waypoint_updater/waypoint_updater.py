#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.ego = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.ego and self.waypoints_tree:
                next_wp = self.get_next_waypoint(self.ego)
                self.publish(next_wp)
            rate.sleep()

    def pose_cb(self, msg):
        self.ego = msg.pose

    def waypoints_cb(self, msg):   
        self.waypoints = msg.waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.waypoints]
            self.waypoints_tree =  KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
    
    def get_next_waypoint(self, ref):
        # get next waypoint after ref
        # Note: does not take into account the orientation or direction of ref
        index = self.waypoints_tree.query([ref.position.x, ref.position.y], 1)[1]
        
        wp_next = self.waypoints_2d[index]
        wp_prev = self.waypoints_2d[index-1]

        # dot product of two vectory points in the same direction is positive
        wp_next_vector = np.array(wp_next)
        wp_prev_vector = np.array(wp_prev)
        ref_vector = np.array([ref.position.x, ref.position.y])

        dot = np.dot( wp_next_vector - wp_prev_vector, ref_vector - wp_next_vector )
        if dot > 0: # wp_next is behind ref 
            index = (index +1) % len(self.waypoints_2d)

        return index
    
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def publish(self, wp_start):
        lane = Lane()
        lane.waypoints = self.waypoints[wp_start:wp_start + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
