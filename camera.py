#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

def camera_publisher():
    rospy.init_node("camera_publisher", anonymous=True)
    pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    bridge = CvBridge()

    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        rospy.logerr("Camera could not be opened.")
        return

    rate = rospy.Rate(30)  # 30 Hz
    rospy.loginfo("Publishing camera feed...")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            try:
                # Convert the frame to a ROS Image message and publish
                img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                pub.publish(img_msg)
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: %s" % e)
        rate.sleep()

def camera_subscriber():
    rospy.init_node("camera_subscriber", anonymous=True)
    rospy.Subscriber("/camera/image_raw", Image, display_feed)
    rospy.spin()

def display_feed(data):
    bridge = CvBridge()
    try:
        # Convert the ROS Image message back to OpenCV format
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        cv2.imshow("Live Camera Feed", cv_image)
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s" % e)

if __name__ == "__main__":
    try:
        mode = raw_input("Enter 'p' to publish or 's' to subscribe: ").strip().lower()
        if mode == 'p':
            camera_publisher()
        elif mode == 's':
            camera_subscriber()
        else:
            print "Invalid mode. Enter 'p' for publisher or 's' for subscriber."
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
