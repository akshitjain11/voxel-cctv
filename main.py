import cv2
import math
import numpy as np
from ultralytics import YOLO
import face_recognition
import os

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Load known faces from folders
known_face_encodings = []
known_face_names = []

known_people_dir = "known_people"  # Folder containing subfolders for each person
for person_name in os.listdir(known_people_dir):
    person_folder = os.path.join(known_people_dir, person_name)
    if os.path.isdir(person_folder):  # Ensure it's a directory
        for filename in os.listdir(person_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_folder, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:  # Ensure encoding was found
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(person_name)  # Use folder name as the person's name

# Camera setup
cap = cv2.VideoCapture(0)

# Load your 2D map image
MAP_IMAGE_PATH = "gamingroomlayout.png"
map_img = cv2.imread(MAP_IMAGE_PATH)
MAP_HEIGHT, MAP_WIDTH = map_img.shape[:2]

# ====== CAMERA ALIGNMENT SETTINGS ======
CAMERA_POS = (760, 100)
CAMERA_ANGLE = 180
SCALE_FACTOR = 100
# =======================================

KNOWN_DISTANCE = 2.0
KNOWN_BOX_AREA = 25000
CAMERA_FOV = 60

label_positions = []  # Keep track of label positions to avoid overlapping

def calculate_distance(box_area):
    return math.sqrt((KNOWN_DISTANCE ** 2 * KNOWN_BOX_AREA) / box_area)

def adjust_label_position(person_x, person_y):
    """Adjust label position to avoid collision with other labels."""
    for existing_x, existing_y in label_positions:
        if abs(existing_x - person_x) < 50 and abs(existing_y - person_y) < 20:
            person_y -= 30  # Shift the label upwards
            break
    label_positions.append((person_x, person_y))
    return person_y

def plot_person_on_map(map_img, x, y, distance, name):
    map_with_overlay = map_img.copy()
    angle_rad = math.radians(CAMERA_ANGLE)
    
    # Convert camera-relative coordinates to map coordinates with rotation
    rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    person_x = CAMERA_POS[0] + int(rotated_x * SCALE_FACTOR)
    person_y = CAMERA_POS[1] - int(rotated_y * SCALE_FACTOR)

    person_x = np.clip(person_x, 0, MAP_WIDTH)
    person_y = np.clip(person_y, 0, MAP_HEIGHT)

    person_y = adjust_label_position(person_x, person_y)  # Adjust label position to avoid collisions

    # Draw a small circle for the person
    cv2.circle(map_with_overlay, (person_x, person_y), 5, (255, 0, 0), -1)

    # Define label position (offset upwards to separate from the person)
    label_x = person_x + 20
    label_y = person_y - 50

    # Draw a line from the person to the label
    cv2.line(map_with_overlay, (person_x, person_y), (label_x, label_y), (0, 0, 255), 1)

    # Draw the label with the person's name and distance
    cv2.putText(map_with_overlay, f"{name} ({distance:.2f}m)", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return map_with_overlay

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=0, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    map_with_overlay = map_img.copy()
    label_positions.clear()  # Reset label positions for each frame

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

        person_frame = frame[y1:y2, x1:x2]
        rgb_person_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_person_frame)
        face_encodings = face_recognition.face_encodings(rgb_person_frame, face_locations)

        name = "Unknown"
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                break

        map_with_overlay = plot_person_on_map(map_with_overlay, x, y, distance, name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({distance:.2f}m)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("2D Map", map_with_overlay)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
