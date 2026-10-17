"""Compare a photo with the owner's enrolled face embedding."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, default=Path("data/owner_embedding.pt"))
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(image_size=160, keep_all=False, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    with Image.open(args.image) as image:
        face = detector(image.convert("RGB"))
    if face is None:
        raise SystemExit("No face detected in the test image.")

    # Only load an embedding file you created or trust.
    owner = torch.load(args.embedding, map_location="cpu", weights_only=True)
    owner = torch.nn.functional.normalize(owner.float().flatten(), dim=0)
    with torch.no_grad():
        query = model(face.unsqueeze(0).to(device)).squeeze(0).cpu()
        query = torch.nn.functional.normalize(query, dim=0)
    similarity = torch.dot(owner, query).item()
    result = "OWNER (candidate)" if similarity >= args.threshold else "UNKNOWN"
    print(f"Cosine similarity: {similarity:.3f}")
    print(f"Threshold: {args.threshold:.3f}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
