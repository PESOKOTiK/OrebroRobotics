#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import math
import time

WAYPOINTS_FILE = '/home/pi/ros2_ws/tmp/waypoints.txt'

class GyroWaypointFollower(Node):
    def __init__(self):
        super().__init__('gyro_waypoint_follower')
        self.pub = self.create_publisher(Twist, '/rp5/cmd', 10)
        self.gyro_angle = 0.0
        self.gyro_ready = False

        # Speeds & tolerances
        self.linear_speed = 0.08     # m/s
        self.angular_speed = 0.7     # rad/s
        self.turn_tol = math.radians(2)

        # Subscribe to gyro (radians)
        self.create_subscription(Float32, '/rp5/rotation', self.gyro_cb, 1)

        # Load waypoints: (x, y, rel_angle_deg)
        self.waypoints = self.load_waypoints(WAYPOINTS_FILE)
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints")

        # Wait for gyro
        self.wait_for_gyro()

        # Compute gyro→map offset
        x0, y0, _ = self.waypoints[0]
        x1, y1, _ = self.waypoints[1]
        desired0 = math.atan2(y1 - y0, x1 - x0)
        self.gyro_offset = self.normalize_ang(desired0 - self.gyro_angle)
        self.get_logger().info(f"Gyro offset: {math.degrees(self.gyro_offset):+.1f}°")

        # Start
        self.follow_path()

    def gyro_cb(self, msg: Float32):
        self.gyro_angle = msg.data
        self.gyro_ready = True

    def wait_for_gyro(self):
        self.get_logger().info("Waiting for gyro...")
        t0 = time.time()
        while not self.gyro_ready and time.time() - t0 < 10:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.gyro_ready:
            self.get_logger().error("Gyro timeout! Exiting.")
            exit(1)
        self.get_logger().info(f"Gyro ready: {math.degrees(self.gyro_angle):.1f}°")

    def load_waypoints(self, path):
        wps = []
        with open(path) as f:
            for line in f:
                gx, gy, x, y, a = line.strip().split(',')
                wps.append((float(x), float(y), float(a)))
        return wps

    def follow_path(self):
        if len(self.waypoints) < 2:
            self.get_logger().error("Need ≥2 waypoints"); return

        # 1) Drive from WP0→WP1
        x0, y0, _ = self.waypoints[0]
        x1, y1, _ = self.waypoints[1]
        d01 = math.hypot(x1-x0, y1-y0)
        self.get_logger().info(f"Drive {d01*100:.1f}cm to WP1")
        self.drive(d01)

        # 2) Each subsequent waypoint: invert rel_angle sign, rotate, then drive
        for i in range(2, len(self.waypoints)):
            # Invert sign here:
            raw_deg = self.waypoints[i-1][2]
            rel_rad = -math.radians(raw_deg)

            self.get_logger().info(f"Rotate {math.degrees(rel_rad):+.1f}° at WP{i}")
            self.rotate(rel_rad)

            x_prev, y_prev, _ = self.waypoints[i-1]
            x_next, y_next, _ = self.waypoints[i]
            dist = math.hypot(x_next-x_prev, y_next-y_prev)
            self.get_logger().info(f"Drive {dist*100:.1f}cm to WP{i+1}")
            self.drive(dist)

        self.stop()
        self.get_logger().info("All waypoints done")

    def rotate(self, rel_rad):
        # Map‐relative headings: gyro + offset
        g = self.get_gyro()
        curr_map = self.normalize_ang(g + self.gyro_offset)
        target = self.normalize_ang(curr_map + rel_rad)

        cmd = Twist()
        # Positive rel_rad → CCW (left), negative → CW (right)
        cmd.angular.z = self.angular_speed * math.copysign(1, rel_rad)

        while True:
            g = self.get_gyro()
            map_h = self.normalize_ang(g + self.gyro_offset)
            err = self.normalize_ang(target - map_h)
            if abs(err) < self.turn_tol:
                break
            self.pub.publish(cmd)
            time.sleep(0.01)

        self.stop()
        time.sleep(0.15)

    def drive(self, distance):
        cmd = Twist(); cmd.linear.x = self.linear_speed
        self.pub.publish(cmd)
        time.sleep(distance / self.linear_speed)
        self.stop()
        time.sleep(0.15)

    def stop(self):
        self.pub.publish(Twist())
        time.sleep(0.05)

    def get_gyro(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        return self.gyro_angle

    @staticmethod
    def normalize_ang(a):
        return (a + math.pi) % (2*math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    GyroWaypointFollower()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
