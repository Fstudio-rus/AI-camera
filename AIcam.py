import cv2
import mediapipe as mp
from ultralytics import YOLO
import time
import numpy as np

# ----------------- YOLOv8 for human detection -----------------
model = YOLO("yolov8n.pt")  # COCO model (0 = person)

# ----------------- Mediapipe Hands -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ----------------- Webcam setup -----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# ----------------- Video writer -----------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ultimate_smart_camera_corrected2.avi', fourcc, 30.0, (frame_width, frame_height))

prev_time = 0
frame_count = 0

# ----------------- Function to count raised fingers -----------------
def count_fingers(hand_landmarks, hand_label):
    tips_ids = [4,8,12,16,20]
    fingers = 0

    # Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
            fingers += 1
    else:
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0]-1].x:
            fingers += 1

    # Other fingers
    for id in range(1,5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id]-2].y:
            fingers += 1

    return fingers

# ----------------- Main loop -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()
    small_frame = cv2.resize(frame, (320, 240))
    frame_count += 1

    # ----------------- YOLO detection -----------------
    if frame_count % 2 == 0:
        results = model(small_frame)
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                if int(cls) == 0:  # person
                    box = box.detach().cpu().numpy() if hasattr(box, "detach") else box
                    x1, y1, x2, y2 = (box * [frame_width/320, frame_height/240, frame_width/320, frame_height/240]).astype(int)
                    cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(annotated_frame, "Person", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    # ----------------- Mediapipe Hands -----------------
    # Draw on the original frame, not the small_frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands_detector.process(rgb_frame)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'

            # ----------------- Draw skeleton (bones) directly on original frame -----------------
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

            # ----------------- Count raised fingers -----------------
            fingers_up = count_fingers(hand_landmarks, label)
            x0 = int(hand_landmarks.landmark[0].x * frame_width)
            y0 = int(hand_landmarks.landmark[0].y * frame_height)
            cv2.putText(annotated_frame, f'{label} Fingers: {fingers_up}', (x0-30, y0-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    # ----------------- Display FPS -----------------
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    # ----------------- Show and save video -----------------
    cv2.imshow("Ultimate Smart Camera", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------- Release resources -----------------
cap.release()
out.release()
cv2.destroyAllWindows()
hands_detector.close()
