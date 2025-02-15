import cv2
import math
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Camera setup
cap = cv2.VideoCapture(0)

# Load your 2D map image
MAP_IMAGE_PATH = "gamingroomlayout.png"
map_img = cv2.imread(MAP_IMAGE_PATH)
MAP_HEIGHT, MAP_WIDTH = map_img.shape[:2]

# ====== NEW: CAMERA ALIGNMENT SETTINGS ======
# 1. Find pixel coordinates of the camera in your PNG using an image editor
CAMERA_POS = (760, 100)  # Example: (x,y) where camera is drawn on your map

# 2. Set camera rotation (0° = facing up, 90° = facing right)
CAMERA_ANGLE = 180  # Degrees (match your camera's orientation in the PNG)

# 3. Set scale (measure a real-world distance in your room and match to map pixels)
SCALE_FACTOR = 100  # 1 meter = 100 pixels in the map image
# ============================================

# Other calibration
KNOWN_DISTANCE = 2.0
KNOWN_BOX_AREA = 25000
CAMERA_FOV = 60


def calculate_distance(box_area):
    return math.sqrt((KNOWN_DISTANCE ** 2 * KNOWN_BOX_AREA) / box_area)


def plot_person_on_map(x, y, distance):
    map_with_overlay = map_img.copy()

    # Convert camera-relative to map coordinates with rotation
    angle_rad = math.radians(CAMERA_ANGLE)
    rotated_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    rotated_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    person_x = CAMERA_POS[0] + int(rotated_x * SCALE_FACTOR)
    person_y = CAMERA_POS[1] - int(rotated_y * SCALE_FACTOR)  # Y-axis inverted

    # Keep within bounds
    person_x = np.clip(person_x, 0, MAP_WIDTH)
    person_y = np.clip(person_y, 0, MAP_HEIGHT)

    # Smaller circle with radius 5 instead of 10
    cv2.circle(map_with_overlay, (person_x, person_y), 5, (255, 0, 0), -1)  # Changed radius to 5
    cv2.putText(map_with_overlay, f"Distance: {distance:.2f}m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return map_with_overlay


while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, classes=0, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        box_area = (x2 - x1) * (y2 - y1)
        if box_area < 100: continue

        distance = calculate_distance(box_area)
        frame_width = frame.shape[1]
        person_center_x = (x1 + x2) / 2
        offset = (person_center_x - frame_width / 2) / (frame_width / 2)
        angle = offset * (CAMERA_FOV / 2)

        x = distance * math.sin(math.radians(angle))
        y = distance * math.cos(math.radians(angle))

        map_with_overlay = plot_person_on_map(x, y, distance)
    else:
        map_with_overlay = map_img.copy()

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("2D Map", map_with_overlay)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()