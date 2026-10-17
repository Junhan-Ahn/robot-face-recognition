"""OpenCV LBPH shared preprocessing: grayscale, Haar detection, 200x200 resize."""
import cv2

def detect_face(image, detector):
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(boxes) == 0:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return cv2.resize(gray[y:y+h, x:x+w], (200, 200))

def load_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("Cannot load Haar Cascade.")
    return detector
