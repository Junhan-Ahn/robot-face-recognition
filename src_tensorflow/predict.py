"""Compare one face to the owner template; educational use only."""
import argparse
import numpy as np
import tensorflow as tf
from common import crop_face, normalize, to_batch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--image",required=True)
    p.add_argument("--model",default="mobilenet_face.keras")
    p.add_argument("--template",default="owner_face.npz")
    args=p.parse_args()
    model=tf.keras.models.load_model(args.model)
    saved=np.load(args.template)
    owner_vector=normalize(saved["owner_vector"])
    threshold=float(saved["threshold"])
    feature=normalize(model.predict(to_batch([crop_face(args.image)]),verbose=0))[0]
    similarity=float(np.dot(feature,owner_vector))
    print(f"Cosine similarity: {similarity:.4f}; demo threshold: {threshold:.2f}")
    print("OWNER candidate" if similarity>=threshold else "UNKNOWN candidate")
    print("Unvalidated face similarity demo; do not use for access control.")

if __name__=="__main__": main()
