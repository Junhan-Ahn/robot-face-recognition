"""Live OpenCV LBPH demo. Q quits; use a trusted local my_face.yml."""
import argparse
import cv2
from common import detect_face, load_detector

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="my_face.yml")
    p.add_argument("--threshold", type=float, default=60.0)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)
    detector = load_detector()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            face = detect_face(frame, detector)
            if face is None:
                text = "NO FACE"
            else:
                label, distance = recognizer.predict(face)
                result = "OWNER" if label == 1 and distance < args.threshold else "UNKNOWN"
                text = f"{result} distance={distance:.1f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.imshow("OpenCV LBPH (Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
