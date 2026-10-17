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
