#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Model configuration
MODEL_CONFIGS = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
ENCODER = 'vits'
CHECKPOINT_PATH = 'checkpoints/depth_anything_v2_vits.pth'

def camera_publisher():
    rospy.init_node("camera_publisher", anonymous=True)
    pub = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
    bridge = CvBridge()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("Camera could not be opened.")
        return

    rate = rospy.Rate(30)
    rospy.loginfo("Publishing camera feed...")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            try:
                img_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
                pub.publish(img_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")
        rate.sleep()

def camera_subscriber():
    rospy.init_node("camera_subscriber", anonymous=True)
    
    # Initialize depth estimation model
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = DepthAnythingV2(**MODEL_CONFIGS[ENCODER])
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    model = model.to(device).eval()
    
    bridge = CvBridge()
    
    def image_callback(msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Perform depth estimation
            with torch.no_grad():
                depth = model.infer_image(cv_image)
            
            # Normalize depth map for visualization
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_normalized = (depth_normalized * 255).astype(np.uint8)
            
            # Display results
            cv2.imshow("Color Image", cv_image)
            cv2.imshow("Depth Map", depth_normalized)
            cv2.waitKey(1)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    rospy.Subscriber("/camera/image_raw", Image, image_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        mode = input("Enter 'p' to publish or 's' to subscribe: ").strip().lower()
        if mode == 'p':
            camera_publisher()
        elif mode == 's':
            camera_subscriber()
        else:
            print("Invalid mode. Enter 'p' for publisher or 's' for subscriber.")
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
