# Robot Owner Face Recognition

Educational AI project with **Python, PyTorch, OpenCV and facenet-pytorch**.

This exercise uses a pretrained FaceNet model to extract face embeddings and compare cosine similarity. It does **not** retrain FaceNet.

## Setup

Install dependencies in a compatible Python environment:

```bash
pip install -r requirements.txt
```

Put 8 consented owner photos in `data/owner/` (e.g. `01.jpg` to `08.jpg`). Enroll:

```bash
python src/enroll.py --input data/owner --output data/owner_embedding.pt
```

Compare a separate test photo:

```bash
python src/recognize.py --image test.jpg --embedding data/owner_embedding.pt
```

A default similarity threshold of 0.70 is **illustrative, not calibrated**. Evaluate with held-out owner/non-owner photos; a wrong match is possible. Do not use this educational demo for real-world security or consequential identity decisions.

## Privacy

Never upload face photos or biometric embeddings to this public repository. The `.gitignore` excludes common image files and `data/`. Get consent before using someone's photos.

## Teaching plan

1. Explain detection, embeddings and cosine similarity.
2. Extract an owner's embedding from multiple photos.
3. Compare a test photo against the owner.
4. Discuss threshold selection, false positives and false negatives.

## Files

- `src/enroll.py` — register owner
- `src/recognize.py` — test photo
- `requirements.txt` — dependencies

## 12-step Google Colab practical (start here)

Open [the 12-step notebook](notebooks/face_recognition_12_steps.ipynb) in Google Colab and run cells in order.

1. Check Python and PyTorch
2. Install packages
3. Import libraries
4. Upload faces.zip
5. Extract and inspect owner photos
6. Initialize MTCNN and FaceNet
7. Detect faces
8. Extract 512-dimensional embeddings
9. Build owner's reference vector
10. Compare a new test photo
11. Save the owner embedding
12. Download and use the Windows webcam demo

**Final source code** lives in `src/enroll.py`, `src/recognize.py`, and `src/webcam.py`.
For Windows webcam use, install `requirements.txt` and run:
```bash
python src/webcam.py --embedding data/owner_embedding.pt
```
Press Q to exit. The project is educational and has not been validated for real-world identity verification.
