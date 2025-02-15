import cv2
import math
import numpy as np
from ultralytics import YOLO
import face_recognition
import os

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Load known faces
known_face_encodings = []
known_face_names = []

known_people_dir = "known_people"
for filename in os.listdir(known_people_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_people_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Camera setup
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Cannot open camera!"

MAP_WIDTH, MAP_HEIGHT = 800, 600
CAMERA_POS = (MAP_WIDTH // 2, MAP_HEIGHT - 50)
KNOWN_DISTANCE = 2.0
KNOWN_BOX_AREA = 25000
CAMERA_FOV = 60

def calculate_distance(box_area):
    return math.sqrt((KNOWN_DISTANCE ** 2 * KNOWN_BOX_AREA) / box_area)

def plot_person_on_map(map_img, x, y, distance, name):
    """Plot a single person on the map image."""
    scale = 100  # 1m = 100 pixels
    person_x = CAMERA_POS[0] + int(x * scale)
    person_y = CAMERA_POS[1] - int(y * scale)
    person_x = np.clip(person_x, 0, MAP_WIDTH)
    person_y = np.clip(person_y, 0, MAP_HEIGHT)
    cv2.circle(map_img, (person_x, person_y), 10, (255, 0, 0), -1)
    cv2.putText(map_img, f"{name} ({distance:.2f}m)", (person_x - 50, person_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=0, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    map_img = np.ones((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8) * 255
    cv2.circle(map_img, CAMERA_POS, 10, (0, 0, 255), -1)  # Draw the camera position

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < 100:
            continue

        distance = calculate_distance(box_area)
        frame_width = frame.shape[1]
        person_center_x = (x1 + x2) / 2
        offset = (person_center_x - frame_width / 2) / (frame_width / 2)
        angle = offset * (CAMERA_FOV / 2)
        x = distance * math.sin(math.radians(angle))
        y = distance * math.cos(math.radians(angle))

        # Extract face from detected person region
        person_frame = frame[y1:y2, x1:x2]
        rgb_person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_person_frame)
        face_encodings = face_recognition.face_encodings(rgb_person_frame, face_locations)

        name = "Unknown"
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                break

        # Plot each person on the map
        plot_person_on_map(map_img, x, y, distance, name)

        # Draw a rectangle around the detected person
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({distance:.2f}m)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("2D Map", map_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
