#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import math
import yaml
from scipy.spatial import KDTree
import os

STATE_COUNT_THRESHOLD = 3
MIN_DIST_TO_LIGHT = 22
MAX_DIST_TO_LIGHT = 100
MAX_IMAGE_TIME_DIFF = 5
MIN_IMAGE_TIME_DIFF = 0.25
MIN_SCORE_THRESHOLD = 0.6

DO_DATA_COLLECTION_SIM = False
DO_DATA_COLLECTION_SITE = False

def dist(x1, y1, x2, y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )

def get_nearest(ref, tree, tree_array):
    #get element from tree which is nearest and in front of ref
    index = tree.query([ref.position.x, ref.position.y], 1)[1]
    
    wp_next = tree_array[index]
    wp_prev = tree_array[index-1]

    # dot product of two vectory points in the same direction is positive
    wp_next_vector = np.array(wp_next)
    wp_prev_vector = np.array(wp_prev)
    ref_vector = np.array([ref.position.x, ref.position.y])

    dot = np.dot( wp_next_vector - wp_prev_vector, ref_vector - wp_next_vector )
    if dot > 0: # wp_next is behind ref 
        index = (index +1) % len(tree_array)

    return index    

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None

        self.lights = None
        self.lights_2d = None
        self.lights_tree = None

        self.light_classifier = None
        
        self.camera_image = None
        self.last_image_time = -999

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']
        self.stop_line_positions_tree = KDTree(self.stop_line_positions)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.config['is_site'])
        self.listener = tf.TransformListener()

        rospy.logwarn("TL intialized")

        self.loop()

    def loop(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            #if no image has arrived, use ground truth data
            if current_time > self.last_image_time + MAX_IMAGE_TIME_DIFF:
                light_wp, light = self.process_traffic_lights()
                if not light:
                    rate.sleep()
                    continue

                rospy.logwarn("no camera image receveied within last %.1fs, using ground truth TL data", MAX_IMAGE_TIME_DIFF)

                self.publish(light_wp, light.state)                

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        if not self.waypoints_tree:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in self.waypoints]
            self.waypoints_tree =  KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if not self.lights_tree:
            self.lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in self.lights]
            self.lights_tree =  KDTree(self.lights_2d)
            

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        if DO_DATA_COLLECTION_SIM or DO_DATA_COLLECTION_SITE:
            self.do_data_collection(msg)
            return

        #process only every N-th image
        current_time = rospy.get_time()
        if current_time < self.last_image_time + MIN_IMAGE_TIME_DIFF:
            return
        
        self.camera_image = msg
        self.last_image_time = rospy.get_time()
        

        light_wp, light = self.process_traffic_lights()
        if not light:
            return

        # get actual light state from camera
        light_state_predicted = self.get_light_state(light)
        if light_state_predicted == TrafficLight.UNKNOWN:
            return

        #if light.state != light_state_predicted:
        #    rospy.logwarn("traffic light prediction error (predicted = %d, ground truth = %d)", light_state_predicted, light.state)

        self.publish(light_wp, light_state_predicted)

    def publish(self, light_wp, state):
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED or state == TrafficLight.YELLOW else -1
            self.last_wp = light_wp            
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.light_classifier or self.last_image_time == 0):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN 

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        prediction, score = self.light_classifier.get_classification(cv_image)
        if score < MIN_SCORE_THRESHOLD:
            return TrafficLight.UNKNOWN 

        rospy.logwarn("predicted: %d (%.3f) / label: %d", prediction, score, light.state)

        return prediction

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and output its
            location and color from the ground truth data. The actual color from the camera will be determined later

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.pose 
        or not self.waypoints 
        or not self.waypoints_tree 
        or not self.stop_line_positions 
        or not self.stop_line_positions_tree
        or not self.lights
        or not self.lights_tree):
            return -1, None

        #find the closest visible traffic light (if one exists)
        light_idx = get_nearest(self.pose.pose, self.lights_tree, self.lights_2d)
        light = self.lights[light_idx]
        distance = dist(light.pose.pose.position.x, light.pose.pose.position.y, self.pose.pose.position.x, self.pose.pose.position.y)
           
        if distance > MIN_DIST_TO_LIGHT and distance < MAX_DIST_TO_LIGHT:
            #find stop line closest to light
            stop_line_idx = self.stop_line_positions_tree.query([light.pose.pose.position.x, light.pose.pose.position.y], 1)[1]
            stop_line =  self.stop_line_positions[stop_line_idx]

            #find waypoint closes to stop line
            wp_idx = self.waypoints_tree.query(stop_line, 1)[1]
            wp = self.waypoints[wp_idx -1] #grab the one behind to make sure we do not pass the line

            return wp_idx, light # we will overwrite the state later if a camera image has arrived
        else:
            return -1, None
        

    def do_data_collection(self, image):
        if DO_DATA_COLLECTION_SIM:
            if self.pose:
                #get nearest light which is in front of ego
                index = get_nearest(self.pose.pose, self.lights_tree, self.lights_2d)
                light = self.lights[index]
                distance = dist(light.pose.pose.position.x, light.pose.pose.position.y,self.pose.pose.position.x, self.pose.pose.position.y)

                #rospy.logwarn("dist = %d", distance)            
                if distance > MIN_DIST_TO_LIGHT and distance < MAX_DIST_TO_LIGHT:
                    label = light.state
                    cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")

                    filename = os.path.join("data", str(label), str(distance) + ".jpg")
                    cv2.imwrite(filename, cv_image)        
                #else:
                #    cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
                #    filename = os.path.join("data", "4", str(distance) + ".jpg")
                #    cv2.imwrite(filename, cv_image)   
        elif DO_DATA_COLLECTION_SITE:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            filename = os.path.join("data", str(rospy.get_time()) + ".jpg")
            cv2.imwrite(filename, cv_image)  

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
