import cv2
from ultralytics import YOLO
import cvzone
import pyttsx3
import threading
from ipcam import cam

# Initialize pyttsx3 for offline text-to-speech
engine = pyttsx3.init()

# Create a lock for thread safety
tts_lock = threading.Lock()

def play_sound(text):
    """ Function to convert text to speech using pyttsx3. """
    with tts_lock:  # Ensure that only one thread can access the TTS engine at a time
        engine.say(text)
        engine.runAndWait()

def play_sound_async(text):
    """ Run play_sound in a separate thread to avoid blocking. """
    thread = threading.Thread(target=play_sound, args=(text,))
    thread.start()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("yolo11s.pt")

# Open the video capture (use webcam)
cap = cv2.VideoCapture('rtsp://192.168.0.102:8080/h264_aac.sdp')

# Set to store already spoken track IDs to avoid repeating
spoken_ids = set()
count=0
while True:
    frame=cam()
    count += 1
    if count % 3 != 0:
        continue

   

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
        
        # Dictionary to count classes based on track IDs for the current frame
        current_frame_counter = {}

        # Iterate through detected objects
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
            
            # Count the object only if it's a new detection
            if track_id not in spoken_ids:
                spoken_ids.add(track_id)
                
                # Increment the count for the detected class
                if c not in current_frame_counter:
                    current_frame_counter[c] = 0
                current_frame_counter[c] += 1

        # Announce the current counts for each detected class
        for class_name, count in current_frame_counter.items():
            if count > 0:  # Only announce if there are detected objects
                count_text = f"{count} {class_name}" if count > 1 else f"One {class_name}"
                play_sound_async(count_text)  # Convert count to speech
                current_frame_counter[class_name] = 0  # Reset count after announcement

    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
