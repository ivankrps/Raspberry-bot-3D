import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.listener_callback, 10)
        self.bridge = CvBridge()

    def listener_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
