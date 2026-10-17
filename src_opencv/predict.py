"""Predict owner/unknown with LBPH. Distance is not a probability."""
import argparse
import cv2
from common import detect_face, load_detector

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="my_face.yml")
    p.add_argument("--threshold", type=float, default=60.0)
    args = p.parse_args()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(args.model)
    face = detect_face(cv2.imread(args.image), load_detector())
    if face is None:
        raise SystemExit("얼굴을 검출하지 못했습니다.")
    label, distance = recognizer.predict(face)
    result = "OWNER" if label == 1 and distance < args.threshold else "UNKNOWN"
    print(f"label={label}, distance={distance:.2f}, result={result}")
    print("distance는 확률이 아니며, threshold=60은 실습용 임시값입니다.")

if __name__ == "__main__":
    main()
