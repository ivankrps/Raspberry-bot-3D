# robot_control_service.py
import os
import csv
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from Raspblock import Raspblock
import cv2
import numpy as np

robot = Raspblock()

class RobotControlServer(Node):
    def __init__(self):
        super().__init__('robot_control_server')

        # --- capture logic state ---
        self.capture_taken = False
        self.prev_speeds = {'x': 0, 'y': 0, 'z': 0}
        self.variation_threshold = 10

        # CSV log setup
        self.log_file = 'capture_log.csv'
        self._ensure_log_file()

        # --- ROS interfaces ---
        self.buzzer_service = self.create_service(
            SetBool, 'buzzer_control', self.handle_buzzer_control
        )
        self.movement_subscription = self.create_subscription(
            Twist, 'movement_control', self.handle_movement, 10
        )

        # --- camera init ---
        self.cap = cv2.VideoCapture(4)
        if not self.cap.isOpened():
            self.get_logger().error("Camera could not be opened.")
            raise RuntimeError("Camera not accessible")

        # Request a higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Read back the actual resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"Camera resolution set to {self.frame_width}x{self.frame_height}"
        )

        # Load calibration if available
        try:
            data = np.load('camera_calibration.npz')
            self.mtx, self.dist = data['mtx'], data['dist']
            self.get_logger().info("Camera calibration loaded.")
        except Exception as e:
            self.get_logger().warn(f"No calibration data: {e}")
            self.mtx = self.dist = None

        self.get_logger().info("Robot control server ready.")

    def _ensure_log_file(self):
        """Create CSV with header if it doesn't exist."""
        if not os.path.isfile(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp_ms', 'x', 'y', 'z', 'filename', 'datetime_iso'
                ])

    def handle_buzzer_control(self, request, response):
        if request.data:
            robot.Buzzer_control(1)
            response.success = True
            response.message = "Buzzer turned on"
        else:
            robot.Buzzer_control(0)
            response.success = True
            response.message = "Buzzer turned off"
        return response

    def handle_movement(self, data: Twist):
        # 1) compute scaled speeds
        speeds = {
            'x': int(data.linear.x  * 15),
            'y': int(data.linear.y  * 15),
            'z': int(data.angular.z * 15),
        }

        # 2) send to robot
        robot.Speed_axis_control(
            speeds['x'], speeds['y'], speeds['z']
        )

        # 3) detect big jump → reset capture flag
        if any(
            abs(speeds[ax] - self.prev_speeds[ax]) > self.variation_threshold
            for ax in ('x','y','z')
        ):
            self.capture_taken = False
            self.get_logger().info(
                f"Variation >{self.variation_threshold} detected, resetting capture flag."
            )

        # 4) if not yet captured → take picture + log
        if not self.capture_taken:
            self.capture_taken = True
            self.get_logger().info("Capturing image…")
            filename, ts_ms, dt_iso = self.capture_image()
            self._log_capture(speeds, filename, ts_ms, dt_iso)

        # 5) update history
        self.prev_speeds = speeds

    def capture_image(self):
        """Grab, undistort, save image; return filename, timestamp, iso-datetime."""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image.")
            return None, None, None

        # Undistort without cropping to preserve full resolution
        if self.mtx is not None and self.dist is not None:
            newc, _ = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist,
                (self.frame_width, self.frame_height),
                1, (self.frame_width, self.frame_height)
            )
            undistorted_frame = cv2.undistort(
                frame, self.mtx, self.dist, None, newc
            )
        else:
            undistorted_frame = frame

        # Save image with timestamps
        ts_ms = int(time.time() * 1000)
        dt_iso = datetime.now().isoformat(timespec='seconds')
        filename = f"movement_capture_{ts_ms}.jpg"
        cv2.imwrite(filename, undistorted_frame)
        self.get_logger().info(f"Saved image: {filename}")

        return filename, ts_ms, dt_iso

    def _log_capture(self, speeds, filename, ts_ms, dt_iso):
        """Append a row to the CSV log."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                ts_ms,
                speeds['x'],
                speeds['y'],
                speeds['z'],
                filename,
                dt_iso
            ])
        self.get_logger().info(f"Logged capture to {self.log_file}")

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down…")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
