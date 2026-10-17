# OpenCV LBPH 최종 실습 소스

Google Colab 실습: [12단계 노트북](../notebooks_opencv/face_recognition_steps.ipynb)

이 폴더는 **OpenCV Haar Cascade + LBPH**만 사용합니다. PyTorch, FaceNet은 사용하지 않습니다.

## 설치
```bash
pip install opencv-contrib-python numpy
```
`opencv-python`과 `opencv-contrib-python`을 동시에 설치하지 마세요.

## 실행
```bash
python src_opencv/train.py --input data/owner --output my_face.yml
python src_opencv/predict.py --image test.jpg --model my_face.yml --threshold 60
python src_opencv/webcam.py --model my_face.yml --threshold 60
```

`faces.zip`에는 `owner/01.jpg` ~ `08.jpg`가 들어갑니다. 학습 결과는 `my_face.yml`로 저장됩니다.

LBPH의 `distance`는 확률이 아니며 일반적으로 낮을수록 더 비슷합니다. `60`은 실습용 시작값일 뿐이며 실제 환경에서는 별도 사진으로 검증해야 합니다.

얼굴사진과 학습모델은 개인정보 보호를 위해 공개 저장소에 올리지 마세요. 실제 보안·출입통제 용도로 사용하지 마세요.
