import cv2

def list_cameras():
    """List available cameras by attempting to open them."""
    print("Scanning for available cameras...")
    available_cameras = []
    for i in range(10):  # Check the first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"No camera found at index {i}")
    if not available_cameras:
        print("No cameras detected.")
    return available_cameras

def main():
    # List available cameras
    available_cameras = list_cameras()

    if not available_cameras:
        print("No cameras available. Exiting.")
        return

    # Ask the user to select a camera index
    try:
        camera_index = int(input(f"Select a camera index from {available_cameras}: "))
        if camera_index not in available_cameras:
            print("Invalid selection. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return

    # Open the selected camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Camera with index {camera_index} could not be opened.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the frame
        cv2.imshow(f"Camera Feed (Index {camera_index})", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
