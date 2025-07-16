import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from mediapipe.framework.formats import landmark_pb2

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Buffers
gesture_queue = deque(maxlen=10)
landmark_buffer = []
BUFFER_SIZE = 5

# Logging file
log_file = open("gesture_log.txt", "w")

# Utility Functions
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

def smooth_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    landmark_buffer.append(coords)
    if len(landmark_buffer) > BUFFER_SIZE:
        landmark_buffer.pop(0)
    avg_coords = np.mean(landmark_buffer, axis=0)
    smoothed = []
    for x, y, z in avg_coords:
        lm = landmark_pb2.NormalizedLandmark()
        lm.x, lm.y, lm.z = x, y, z
        smoothed.append(lm)
    return smoothed

def finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

def recognize_gesture(landmarks, handedness):
    fingers = {
        "thumb": False,
        "index": finger_extended(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        "middle": finger_extended(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        "ring": finger_extended(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        "pinky": finger_extended(landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    }

    # Thumb logic adjusted for handedness
    if handedness == "Right":
        fingers["thumb"] = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x
    else:
        fingers["thumb"] = landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.THUMB_IP].x

    if all(fingers.values()):
        return "HELLO"
    if fingers["index"] and not fingers["middle"] and fingers["pinky"]:
        return "I LOVE YOU"
    if not any(fingers.values()):
        return "YES"
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"] and not fingers["thumb"]:
        return "STOP"
    return None

# Start video capture
cap = cv2.VideoCapture(0)
prev_gesture = None
prev_time = time.time()

print("Starting gesture recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    if result.multi_hand_landmarks and result.multi_handedness:
        for lm, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
            if len(lm.landmark) == 21:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                smoothed = smooth_landmarks(lm.landmark)
                handedness = hand_label.classification[0].label  # "Right" or "Left"
                gesture = recognize_gesture(smoothed, handedness)

                gesture_queue.append(gesture)
                if gesture_queue.count(gesture) > len(gesture_queue) // 2 and gesture:
                    prev_gesture = gesture
                    log_file.write(f'{gesture} at {time.strftime("%H:%M:%S")}\n')

                if prev_gesture:
                    cv2.putText(frame, f'Gesture: {prev_gesture}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("ISL Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting gesture recognition...")
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()