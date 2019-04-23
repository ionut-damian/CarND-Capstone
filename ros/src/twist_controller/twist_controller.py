from pid import PID
from yaw_controller import YawController

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, 
                decel_limit, accel_limit, wheel_radius, 
                wheel_base, steer_ratio, max_lat_accel, 
                max_steer_angle):        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        self.vehicle_mass = vehicle_mass + fuel_capacity * GAS_DENSITY
        self.brake_deadband = brake_deadband 
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius  = wheel_radius 
        
        self.vel_pid = PID(0.3, 0.001, 3, 0, 1)
        #self.steer_pid = PID(0.2, 0.0001, 1.0)

        self.time = rospy.get_time()

    def control(self, dbw_enabled, current_vel, target_linear_vel, target_angular_vel):

        if not dbw_enabled:
            self.vel_pid.reset()
            return 0, 0, 0

        current_time = rospy.get_time()

        #throttle
        vel_error = target_linear_vel - current_vel
        throttle = self.vel_pid.step(vel_error, self.time - current_time)

        #steering
        #steer_error = target_angular_vel
        #angle = -1 * self.steer_pid.step(steer_error, self.time - current_time)
        steering_value = self.yaw_controller.get_steering(target_linear_vel, target_angular_vel, current_vel)
        #rospy.logwarn("steer: %.2f %.2f %.2f", target_angular_vel, angle, steering_value)

        #breaks
        break_force = 0

        if target_linear_vel == 0 and current_vel < 0.1:
            throttle = 0
            break_force = 700

        if target_linear_vel < current_vel:
            throttle = 0
            decel = max(target_linear_vel - current_vel, self.decel_limit)
            break_force = abs(decel) * self.vehicle_mass * self.wheel_radius

        self.time = current_time

        return throttle, break_force, steering_value
