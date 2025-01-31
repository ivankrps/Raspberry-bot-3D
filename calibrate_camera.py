import cv2
import numpy as np
import glob

def calibrate_camera(image_folder, chessboard_size=(9, 6)):
    # Prepare object points for a chessboard
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Load all calibration images
    images = glob.glob(f"{image_folder}/*.jpg")
    if not images:
        print("No images found in the folder.")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)

            # Refine corner location and add to image points
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Calibration successful!" if ret else "Calibration failed.")
    print("\nCamera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)

    # Save calibration data
    np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\nCalibration data saved to 'camera_calibration.npz'.")

def main():
    # Change this to the folder containing your calibration images
    image_folder = "calibration_images"
    calibrate_camera(image_folder)

if __name__ == "__main__":
    main()
