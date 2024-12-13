# joystick_buzzer_control.py
import rospy
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist
import pygame

def select_controller():
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()

    if joystick_count == 0:
        print("No joystick detected.")
        return None

    print("Available Controllers:")
    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        print("{}: {}".format(i, joystick.get_name()))

    selected_index = int(raw_input("Select the controller number: "))  # raw_input for Python 2.7
    if selected_index < 0 or selected_index >= joystick_count:
        print("Invalid selection.")
        return None

    selected_joystick = pygame.joystick.Joystick(selected_index)
    selected_joystick.init()
    print("Selected controller: {}".format(selected_joystick.get_name()))
    return selected_joystick

# joystick_buzzer_control.py
def joystick_control():
    joystick = select_controller()
    if joystick is None:
        print("Exiting program.")
        return

    rospy.init_node('joystick_control_client')
    rospy.wait_for_service('buzzer_control')
    buzzer_service = rospy.ServiceProxy('buzzer_control', SetBool)
    movement_pub = rospy.Publisher('/movement_control', Twist, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        pygame.event.pump()

        # Get axis values
        axis_x = joystick.get_axis(0) - 0.00390625
        axis_y = joystick.get_axis(1) - 0.00390625
        axis_z = joystick.get_axis(2)   - 0.00390625

        rospy.loginfo("Publishing movement: X={}, Y={}, Z={}".format(axis_x, axis_y, axis_z))

        # Publish movement commands
        twist = Twist()
        twist.linear.x = axis_x
        twist.linear.y = axis_y
        twist.angular.z = axis_z
        try:
            movement_pub.publish(twist)
        except:
            rospy.logerr("fallo")

        # Control buzzer
        button_buzzer = joystick.get_button(0)
        try:
            buzzer_service(button_buzzer)  # Call the buzzer service
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

        rate.sleep()




if __name__ == "__main__":
    try:
        joystick_control()
    except rospy.ROSInterruptException:
        pass
    finally:
        pygame.quit()
