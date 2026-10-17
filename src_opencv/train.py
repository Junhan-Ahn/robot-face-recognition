"""Train OpenCV LBPH from consented owner photos. Example: python src_opencv/train.py --input data/owner"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from common import detect_face, load_detector

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/owner"))
    p.add_argument("--output", type=Path, default=Path("my_face.yml"))
    args = p.parse_args()
    detector = load_detector()
    faces = []
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        face = detect_face(cv2.imread(str(path)), detector)
        if face is not None:
            faces.append(face)
            print("검출 완료:", path.name)
        else:
            print("검출 실패:", path.name)
    if not faces:
        raise SystemExit("학습 가능한 얼굴 사진이 없습니다.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.full(len(faces), 1, dtype=np.int32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(args.output))
    print(f"학습 완료: {len(faces)}장, 저장: {args.output}")

if __name__ == "__main__":
    main()
