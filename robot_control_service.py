# robot_control_service.py
import rospy
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import Twist
from Raspblock import Raspblock

robot = Raspblock()

# Handle the buzzer
def handle_buzzer_control(request):
    # Log in the callback
    #rospy.loginfo("Received message in handle_movement: {}".format(request))

    if request.data:
        robot.Buzzer_control(1)  # Turn buzzer on
        return SetBoolResponse(success=True, message="Buzzer turned on")
    else:
        robot.Buzzer_control(0)  # Turn buzzer off
        return SetBoolResponse(success=True, message="Buzzer turned off")

# Handle movement commands
# robot_control_service.py
def handle_movement(data):
    # Log the incoming data for debugging
    rospy.loginfo("Received movement: X={}, Y={}, Z={}".format(data.linear.x, data.linear.y, data.angular.z))

    # Assuming `linear.x` and `linear.y` are for movement and `angular.z` is for spin
    Speed_axis_X = int(data.linear.x * 15)
    Speed_axis_Y = int(data.linear.y * 15)
    Speed_axis_Z = int(data.angular.z * 15)

    # Send movement commands to the robot
    robot.Speed_axis_Yawhold_control(Speed_axis_X, Speed_axis_Y)
    rospy.loginfo("Executing movement: X={}, Y={}, Z={}".format(Speed_axis_X, Speed_axis_Y, Speed_axis_Z))


def control_server():
    rospy.init_node('robot_control_server')

    # Buzzer control service
    rospy.Service('buzzer_control', SetBool, handle_buzzer_control)

    # Movement control subscriber
    rospy.Subscriber('/movement_control', Twist, handle_movement)


    rospy.loginfo("Robot control server is ready.")
    rospy.spin()

if __name__ == "__main__":
    try:
        control_server()
    except rospy.ROSInterruptException:
        pass
