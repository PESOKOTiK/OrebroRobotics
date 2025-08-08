#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from aruco_interfaces.msg import ArucoMarkers  # adjust if your msg type differs
import numpy as np
import heapq
import cv2
import os
import time
from collections import deque
import threading

class MazeNavigator(Node):
    def __init__(self):
        super().__init__('maze_navigator')

        # ---- USER SETTINGS ----
        self.goal_id = 15                      # Goal tag ID (change if needed)
        self.robot_id = 5                      # Robot tag ID
        self.save_folder = '/home/pi/ros2_ws/tmp'
        self.image_scale = 2                   # Upscale factor for output image
        self.smooth_frames = 5                 # Tag position smoothing over N frames
        self.tag_grid_offset_x = -30          # Grid X offset for tag-to-map calibration
        self.tag_grid_offset_y = 35            # Grid Y offset for tag-to-map calibration
        self.wall_inflation_radius = 4         # Number of grid cells to inflate around walls

        # ---- State ----
        self.grid = None
        self.map_info = None
        self.tag_history = {}                  # id: deque of last N (x, y)
        self.last_tag_position = {}            # id: (x, y)
        self.path = None                       # Current path (list of (gx, gy))
        self.waypoints = None                  # Waypoints at 90 deg turns

        # ---- Prepare QoS profile ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ---- Subscriptions ----
        self.create_subscription(
            OccupancyGrid,
            '/occupancy/map/grid',
            self.occupancy_callback,
            qos)

        self.create_subscription(
            ArucoMarkers,
            '/aruco/markers/transformed',
            self.aruco_callback,
            qos)

        # ---- Timer: Run loop every 2 seconds ----
        self.create_timer(2.0, self.run_navigation)

        # Ensure save directory exists
        os.makedirs(self.save_folder, exist_ok=True)

        self.get_logger().info("MazeNavigator node started with wall inflation radius = %d cells" % self.wall_inflation_radius)

        # Save an initial debug map after a few seconds (start view)
        threading.Timer(3.0, self.save_startup_image).start()

    def save_startup_image(self):
        if self.grid is None or self.map_info is None:
            self.get_logger().info("No map yet for startup image.")
            return
        grid = self.grid * 255
        h, w = grid.shape
        img = np.stack([grid]*3, axis=-1).astype(np.uint8)
        img = cv2.resize(img, (w*self.image_scale, h*self.image_scale), interpolation=cv2.INTER_NEAREST)

        cv2.circle(img, self.xy_to_img(0,0), 10, (200,0,200), -1)
        cv2.putText(img, "Origin(0,0)", self.xy_to_img(0,0, dy=-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,200), 2)

        for tag_id, history in self.tag_history.items():
            gx, gy = np.median(np.array(history), axis=0).astype(int)
            color = (200,200,200)
            if tag_id == self.robot_id: color = (0,255,0)
            if tag_id == self.goal_id: color = (255,0,0)
            cv2.circle(img, self.xy_to_img(gx, gy), 8, color, 2)
            cv2.putText(img, str(tag_id), self.xy_to_img(gx, gy, dy=-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        img_file = f"{self.save_folder}/startup_debug.png"
        cv2.imwrite(img_file, img)
        self.get_logger().info(f"Startup debug image saved: {img_file}")

    def occupancy_callback(self, msg):
        self.map_info = msg.info
        w, h = msg.info.width, msg.info.height
        arr = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.grid = (arr < 0).astype(np.uint8)

    def aruco_callback(self, msg):
        for m in msg.markers:
            gx, gy = self.world_to_grid(m.pose.position.x, m.pose.position.y)
            if m.id not in self.tag_history:
                self.tag_history[m.id] = deque(maxlen=self.smooth_frames)
            self.tag_history[m.id].append((gx, gy))
            self.last_tag_position[m.id] = (gx, gy)

    def world_to_grid(self, x, y):
        if not self.map_info:
            return None, None
        gx = int(round((x - self.map_info.origin.position.x) / self.map_info.resolution)) + self.tag_grid_offset_x
        gy = int(round((y - self.map_info.origin.position.y) / self.map_info.resolution)) + self.tag_grid_offset_y
        gx = np.clip(gx, 0, self.map_info.width - 1)
        gy = np.clip(gy, 0, self.map_info.height - 1)
        return gx, gy

    def grid_to_world(self, gx, gy):
        if not self.map_info:
            return None, None
        x = gx * self.map_info.resolution + self.map_info.origin.position.x
        y = gy * self.map_info.resolution + self.map_info.origin.position.y
        return x, y

    def inflate_grid(self, grid, radius):
        if radius <= 0:
            return grid.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        return cv2.dilate(grid.astype(np.uint8), kernel)

    def run_navigation(self):
        if self.grid is None or self.map_info is None:
            self.get_logger().info("Waiting for map...")
            return

        robot_xy = self.get_smoothed_tag(self.robot_id)
        goal_xy  = self.get_smoothed_tag(self.goal_id)
        if robot_xy is None or goal_xy is None:
            self.get_logger().warning("Robot or goal tag not detected (using memory if available).")
            return

        path = self.plan_path(robot_xy, goal_xy)
        if path is None:
            self.get_logger().warning("No path found!")
            return

        waypoints = self.extract_waypoints(path)

        img_file = f"{self.save_folder}/maze_debug.png"
        wpt_file = f"{self.save_folder}/waypoints.txt"
        self.save_debug_image(img_file, path, waypoints, robot_xy, goal_xy)
        self.save_waypoints_relative(wpt_file, waypoints)

        self.get_logger().info(f"Path planned with inflation, image saved: {img_file}, waypoints saved: {wpt_file}")

    def get_smoothed_tag(self, tag_id):
        if tag_id in self.tag_history and len(self.tag_history[tag_id]) > 0:
            arr = np.array(self.tag_history[tag_id])
            gx, gy = np.median(arr, axis=0).astype(int)
            return (gx, gy)
        elif tag_id in self.last_tag_position:
            return self.last_tag_position[tag_id]
        else:
            return None

    def plan_path(self, start, goal):
        orig = self.grid.copy()
        sx, sy = start
        gx, gy = goal
        if orig[sy, sx] or orig[gy, gx]:
            self.get_logger().error("Start or goal in wall!")
            return None

        grid = self.inflate_grid(orig, self.wall_inflation_radius)

        h, w = grid.shape
        open_set = []
        heapq.heappush(open_set, (0 + abs(sx - gx) + abs(sy - gy), 0, (sx, sy), []))
        visited = set()
        dirs = [(-1,0), (1,0), (0,-1), (0,1)]

        while open_set:
            f, cost, (x, y), path = heapq.heappop(open_set)
            if (x, y) == (gx, gy):
                return path + [(x, y)]
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not grid[ny, nx]:
                    heapq.heappush(open_set, (cost + 1 + abs(nx - gx) + abs(ny - gy), cost + 1, (nx, ny), path + [(x, y)]))
        return None

    def extract_waypoints(self, path):
        if not path or len(path) < 2:
            return path
        waypoints = [path[0]]
        dx_prev, dy_prev = path[1][0] - path[0][0], path[1][1] - path[0][1]
        for i in range(1, len(path)-1):
            dx, dy = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
            if (dx, dy) != (dx_prev, dy_prev):
                waypoints.append(path[i])
            dx_prev, dy_prev = dx, dy
        waypoints.append(path[-1])
        return waypoints

    def compute_waypoint_orientations(self, waypoints):
        result = []
        for i in range(len(waypoints)):
            gx, gy = waypoints[i]
            if i < len(waypoints)-1:
                nx, ny = waypoints[i+1]
                dx, dy = nx - gx, ny - gy
                if dx > 0:      angle = 0
                elif dx < 0:    angle = 180
                elif dy > 0:    angle = 90
                elif dy < 0:    angle = -90
                else:           angle = 0
            else:
                angle = result[-1][2] if result else 0
            result.append((gx, gy, angle))
        return result

    def save_waypoints_relative(self, filename, waypoints):
        wp_with_abs = self.compute_waypoint_orientations(waypoints)
        result = []
        prev_angle = wp_with_abs[0][2]
        for i, (gx, gy, abs_angle) in enumerate(wp_with_abs):
            rel_angle = 0 if i == 0 else self.normalize_angle_deg(abs_angle - prev_angle)
            x, y = self.grid_to_world(gx, gy)
            result.append((gx, gy, x, y, rel_angle))
            prev_angle = abs_angle
        with open(filename, "w") as f:
            for gx, gy, x, y, rel_angle in result:
                f.write(f"{gx},{gy},{x:.3f},{y:.3f},{rel_angle:.0f}\n")

    @staticmethod
    def normalize_angle_deg(angle):
        return (angle + 180) % 360 - 180

    def save_debug_image(self, filename, path, waypoints, robot_xy, goal_xy):
        grid_vis = self.grid * 255
        h, w = grid_vis.shape
        img = np.stack([grid_vis]*3, axis=-1).astype(np.uint8)
        img = cv2.resize(img, (w*self.image_scale, h*self.image_scale), interpolation=cv2.INTER_NEAREST)

        # Inflate and overlay inflated zones at image resolution
        inflated = self.inflate_grid(self.grid, self.wall_inflation_radius).astype(np.uint8) * 255
        inflated_resized = cv2.resize(inflated, (w*self.image_scale, h*self.image_scale), interpolation=cv2.INTER_NEAREST)
        mask = inflated_resized > 0
        img[mask] = [0, 0, 200]

        # Draw path
        for i in range(1, len(path)):
            x0, y0 = path[i-1]
            x1, y1 = path[i]
            cv2.line(img, self.xy_to_img(x0, y0), self.xy_to_img(x1, y1), (0,0,255), 2)

        # Draw waypoints
        for i, (x, y) in enumerate(waypoints):
            color = (0,255,0) if i == 0 else (255,0,0) if i == len(waypoints)-1 else (0,255,255)
            cv2.circle(img, self.xy_to_img(x, y), 6, color, -1)

        # Draw tags
        for tag_id, history in self.tag_history.items():
            gx, gy = np.median(np.array(history), axis=0).astype(int)
            color = (200,200,200)
            if tag_id == self.robot_id: color = (0,255,0)
            if tag_id == self.goal_id: color = (255,0,0)
            cv2.circle(img, self.xy_to_img(gx, gy), 8, color, 2)
            cv2.putText(img, str(tag_id), self.xy_to_img(gx, gy, dy=-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imwrite(filename, img)

    def xy_to_img(self, gx, gy, dx=0, dy=0):
        x = gx * self.image_scale + self.image_scale//2 + dx
        y = gy * self.image_scale + self.image_scale//2 + dy
        return int(x), int(y)

def main(args=None):
    rclpy.init(args=args)
    node = MazeNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
