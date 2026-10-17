"""Build an owner template; frozen pretrained CNN, no weight training."""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import crop_face, normalize, to_batch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="owner_face.npz")
    p.add_argument("--model",default="mobilenet_face.keras")
    p.add_argument("--threshold",type=float,default=0.85)
    args=p.parse_args()
    paths=sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    if not paths: raise ValueError("No owner photos")
    faces=np.stack([crop_face(p) for p in paths])
    augmentation=tf.keras.Sequential([tf.keras.layers.RandomRotation(.05),tf.keras.layers.RandomTranslation(.05,.05),tf.keras.layers.RandomZoom(.1),tf.keras.layers.RandomContrast(.1)])
    aug=[]
    for face in faces:
        batch=tf.expand_dims(tf.cast(face,tf.float32),0)
        for _ in range(5): aug.append(augmentation(batch,training=True)[0].numpy())
    data=np.concatenate([faces.astype(np.float32),np.asarray(aug,dtype=np.float32)])
    model=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights="imagenet",pooling="avg")
    model.trainable=False
    features=normalize(model.predict(to_batch(data),batch_size=16,verbose=0))
    owner_vector=normalize(features.mean(axis=0))
    model.save(args.model)
    np.savez(args.output,owner_vector=owner_vector,threshold=args.threshold)
    print(f"Saved template from {len(faces)} originals + {len(aug)} augmentations")
    print("WARNING: threshold unvalidated; this is not identity authentication")

if __name__=="__main__": main()
