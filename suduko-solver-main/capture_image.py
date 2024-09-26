import cv2
import os

def capture_images():
    # Initialize the camera (0 is usually the default camera)
    cam = cv2.VideoCapture(0)

    # Create a window named "Smile"
    cv2.namedWindow("Smile")

    # Specify the directory to save captured images
    save_dir = "photos"

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_counter = 0

    # Start an infinite loop to continuously capture frames
    while True:
        ret, frame = cam.read()  # Read the current frame from the camera
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the frame in the "Smile" window
        cv2.imshow("Smile", frame)

        # Wait for 1 ms for a key press
        k = cv2.waitKey(1)

        if k % 256 == 27:  # ESC key (ASCII code 27)
            print("ESC pressed, exiting...")
            break
        elif k % 256 == 32:  # Space key (ASCII code 32)
            # Construct the full path for the image
            img_name = os.path.join(save_dir, "opencv_frame_{}.png".format(img_counter))
            cv2.imwrite(img_name, frame)  # Save the current frame as an image
            print(f"Photo captured: {img_name}")
            img_counter += 1

    # Release the camera and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

# Ensure this code only runs when this file is executed directly
if __name__ == "__main__":
    capture_images()
