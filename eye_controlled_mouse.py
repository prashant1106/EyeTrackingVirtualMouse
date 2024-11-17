import cv2
import dlib
import pyautogui
import numpy as np
from scipy.spatial import distance as dist
import time

# Constants
EYE_AR_THRESH = 0.25  # Threshold for blink detection
EYE_AR_CONSEC_FRAMES = 3  # Consecutive frames for detecting a blink
RIGHT_CLICK_TIME_THRESHOLD = 2  # Time in seconds for detecting right-click
FRAME_SKIP = 5  # Skip every 5th frame for face detection
SCREEN_EDGE_RATIO = 0.2  # Ratio to determine screen edge

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is present
except Exception as e:
    print(f"Error loading predictor model: {e}")
    exit()

def eye_aspect_ratio(eye):
    """Compute the eye aspect ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def control_mouse(eye_left, eye_right, screen_width, screen_height, last_position, movement_threshold=10):
    """Control mouse based on eye movement."""
    left_eye_center = np.mean(eye_left, axis=0)
    right_eye_center = np.mean(eye_right, axis=0)
    eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)

    x, y = eye_center
    x = int(np.interp(x, [0, 640], [0, screen_width]))
    y = int(np.interp(y, [0, 480], [0, screen_height]))

    if abs(x - last_position[0]) > movement_threshold or abs(y - last_position[1]) > movement_threshold:
        pyautogui.moveTo(x, y)
        return (x, y)
    return last_position

def check_screen_edge(eye_left, eye_right, screen_width):
    """Check if the user is looking at the left or right side of the screen."""
    left_eye_center = np.mean(eye_left, axis=0)
    right_eye_center = np.mean(eye_right, axis=0)
    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2

    if eye_center_x < screen_width * SCREEN_EDGE_RATIO:
        return "left"
    elif eye_center_x > screen_width * (1 - SCREEN_EDGE_RATIO):
        return "right"
    return None

def main():
    """Main function to run the eye tracking and mouse control."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    screen_width, screen_height = pyautogui.size()

    blink_counter = 0
    left_right_edge_time = {"left": 0, "right": 0}
    last_position = (0, 0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Invert (flip) the frame horizontally
        frame = cv2.flip(frame, 1)  # 1 means flipping around the vertical axis (horizontal flip)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % FRAME_SKIP == 0:
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

                cv2.polylines(frame, [np.array(left_eye)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.polylines(frame, [np.array(right_eye)], isClosed=True, color=(0, 255, 0), thickness=2)

                last_position = control_mouse(left_eye, right_eye, screen_width, screen_height, last_position)

                screen_edge = check_screen_edge(left_eye, right_eye, screen_width)
                current_time = time.time()

                if screen_edge:
                    if screen_edge == "left":
                        if left_right_edge_time["left"] == 0:
                            left_right_edge_time["left"] = current_time
                        elif current_time - left_right_edge_time["left"] >= RIGHT_CLICK_TIME_THRESHOLD:
                            pyautogui.rightClick()
                            left_right_edge_time["left"] = 0
                    elif screen_edge == "right":
                        if left_right_edge_time["right"] == 0:
                            left_right_edge_time["right"] = current_time
                        elif current_time - left_right_edge_time["right"] >= RIGHT_CLICK_TIME_THRESHOLD:
                            pyautogui.rightClick()
                            left_right_edge_time["right"] = 0

                ear_left = eye_aspect_ratio(left_eye)
                ear_right = eye_aspect_ratio(right_eye)
                ear_avg = (ear_left + ear_right) / 2.0

                if ear_avg < EYE_AR_THRESH:
                    blink_counter += 1
                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                        pyautogui.click()
                else:
                    blink_counter = 0

        frame_count += 1
        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()