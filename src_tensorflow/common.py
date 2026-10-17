import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def normalize(v):
    v=np.asarray(v,dtype=np.float32)
    return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-10)

def crop_face(path):
    image=cv2.imread(str(path))
    if image is None: raise ValueError(f"Cannot read {path}")
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    boxes=detector.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),scaleFactor=1.1,minNeighbors=5,minSize=(40,40))
    if len(boxes)==0: raise ValueError("No face detected")
    x,y,w,h=max(boxes,key=lambda b:b[2]*b[3])
    return cv2.resize(cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2RGB),(224,224))

def to_batch(faces):
    return preprocess_input(np.asarray(faces,dtype=np.float32).copy())
