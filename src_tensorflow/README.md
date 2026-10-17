# TensorFlow / Keras 얼굴 특징 추출 실습

[Google Colab 12단계 노트북](../notebook_tensorflow/face_recognition_12_steps.ipynb)

OpenCV Haar Cascade로 얼굴을 검출하고, **사전학습된 MobileNetV2 CNN의 특징 벡터를 추출**하여 주인 대표 벡터와 코사인 유사도를 비교합니다. CNN 가중치를 새로 학습하는 프로젝트는 아닙니다.

## 실습
- 개인 사진은 GitHub에 업로드하지 않고, `faces.zip` (`owner/01.jpg` ~ `08.jpg`)을 Colab에 직접 업로드합니다.
- 로컬: `pip install -r src_tensorflow/requirements.txt`
- `python src_tensorflow/enroll.py --input data/owner`
- `python src_tensorflow/predict.py --image test.jpg`
- 생성 파일: `mobilenet_face.keras`, `owner_face.npz`

기본 임계값 0.85는 **검증되지 않은 임시 값**입니다. ImageNet MobileNetV2는 얼굴인식 전용 모델이 아니며, 학습에 사용한 사진의 점수로 정확도를 주장할 수 없습니다. 학습에 쓰지 않은 주인 및 타인 사진으로 평가해야 합니다. 얼굴 사진/특징 벡터/모델 파일을 공개 저장소에 올리지 마세요. 보안·출입통제 목적으로 사용하지 마세요.
