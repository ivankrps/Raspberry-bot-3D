import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import pygame

class JoystickTwistPublisher(Node):
    def __init__(self):
        super().__init__('joystick_twist_publisher')

        # Create a publisher for Twist messages
        self.publisher_ = self.create_publisher(Twist, '/movement_control', 1)
        self.timer = self.create_timer(0.1, self.publish_twist)  # Publish at 10 Hz

        # Initialize pygame for joystick input
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            self.get_logger().error("No joystick detected! Please connect a joystick.")
            raise RuntimeError("No joystick connected.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.get_logger().info(f"Joystick initialized: {self.joystick.get_name()}")

        # Twist message to send
        self.twist_msg = Twist()

    def publish_twist(self):
        # Process joystick events
        pygame.event.pump()

        # Read joystick axes
        # Assuming left stick for linear and right stick for angular:
        # Axes 0, 1: Left stick (x, y); Axes 2, 3: Right stick (x, y)
        self.twist_msg.linear.x = self.joystick.get_axis(0)  # Forward/backward
        self.twist_msg.linear.y = -self.joystick.get_axis(1)   # Left/right
        self.twist_msg.angular.z = self.joystick.get_axis(2)  # Spin

        # Log and publish the twist message
        #self.get_logger().info(f"Publishing: linear.x={self.twist_msg.linear.x:.2f}, "
        #                       f"linear.y={self.twist_msg.linear.y:.2f}, "
        #                       f"angular.z={self.twist_msg.angular.z:.2f}")
        self.publisher_.publish(self.twist_msg)

def main(args=None):
    rclpy.init(args=args)

    try:
        node = JoystickTwistPublisher()
        rclpy.spin(node)
    except RuntimeError as e:
        print(e)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        pygame.quit()

if __name__ == '__main__':
    main()
