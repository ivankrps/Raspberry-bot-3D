# robot_control_service.py
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
from Raspblock import Raspblock

robot = Raspblock()

class RobotControlServer(Node):
    def __init__(self):
        super().__init__('robot_control_server')

        # Buzzer control service
        self.buzzer_service = self.create_service(SetBool, 'buzzer_control', self.handle_buzzer_control)

        # Movement control subscriber
        self.movement_subscription = self.create_subscription(
            Twist,
            'movement_control',
            self.handle_movement,
            10
        )

        self.get_logger().info("Robot control server is ready.")

    def handle_buzzer_control(self, request, response):
        """Handle the buzzer control service."""
        if request.data:
            robot.Buzzer_control(1)  # Turn buzzer on
            response.success = True
            response.message = "Buzzer turned on"
        else:
            robot.Buzzer_control(0)  # Turn buzzer off
            response.success = True
            response.message = "Buzzer turned off"
        return response

    def handle_movement(self, data):
        """Handle movement control commands."""
        self.get_logger().info(f"Received movement: X={data.linear.x}, Y={data.linear.y}, Z={data.angular.z}")

        # Assuming `linear.x` and `linear.y` are for movement and `angular.z` is for spin
        Speed_axis_X = int(data.linear.x * 15)
        Speed_axis_Y = int(data.linear.y * 15)
        Speed_axis_Z = int(data.angular.z * 15)

        # Send movement commands to the robot
        robot.Speed_axis_Yawhold_control(Speed_axis_X, Speed_axis_Y)
        self.get_logger().info(f"Executing movement: X={Speed_axis_X}, Y={Speed_axis_Y}, Z={Speed_axis_Z}")

def main(args=None):
    rclpy.init(args=args)

    # Create and spin the server node
    robot_control_server = RobotControlServer()

    try:
        rclpy.spin(robot_control_server)
    except KeyboardInterrupt:
        robot_control_server.get_logger().info("Shutting down robot control server.")
    finally:
        robot_control_server.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
