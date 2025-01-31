import cv2
import numpy as np

# Load calibration data
data = np.load('camera_calibration.npz')
mtx = data['mtx']
dist = data['dist']

# Open the camera feed
cap = cv2.VideoCapture(6)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Undistort the frame
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Display the frames
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
