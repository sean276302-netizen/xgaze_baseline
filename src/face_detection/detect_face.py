import cv2
import dlib
import logging
from contextlib import contextmanager

@contextmanager
def suppress_logging():
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logging.getLogger("ultralytics").setLevel(logging.INFO)

def detect_face_dlib(frame, face_detector):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(small_frame, 1)
    faces = [dlib.rectangle(int(d.left() * 4), int(d.top() * 4), 
                            int(d.right() * 4), int(d.bottom() * 4)) for d in detected_faces]
    return faces

def detect_face_yolo(frame, yolo):
    with suppress_logging():
        detected_faces = []
        results = yolo.predict(frame, conf=0.25, imgsz=640)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detected_faces.append(dlib.rectangle(x1, y1, x2, y2))
    return detected_faces

def detect_face_yolo_v2(frame, yolo):
    with suppress_logging():
        detected_faces = []
        detected_faces_for_crop_head = []
        results = yolo.predict(frame, conf=0.25, imgsz=640)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detected_faces.append(dlib.rectangle(x1, y1, x2, y2))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detected_faces_for_crop_head.append([x1, y1, x2, y2])
    return detected_faces, detected_faces_for_crop_head

def detect_face_yolo_for_face_model(frame, yolo):
    with suppress_logging():
        detected_faces = []
        results = yolo.predict(frame, conf=0.25, imgsz=640)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detected_faces.append([x1, y1, x2, y2])
    return detected_faces