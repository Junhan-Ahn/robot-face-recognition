"""Create a reference embedding from consented owner photos."""
import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN, InceptionResnetV1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/owner"))
    parser.add_argument("--output", type=Path, default=Path("data/owner_embedding.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    vectors = []
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for path in paths:
        try:
            with Image.open(path) as image:
                face = detector(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            print(f"Skipped {path.name}: {exc}")
            continue
        if face is None:
            print(f"No face detected: {path.name}")
            continue
        with torch.no_grad():
            vector = model(face.unsqueeze(0).to(device)).squeeze(0).cpu()
            vectors.append(torch.nn.functional.normalize(vector, dim=0))
        print(f"Processed {path.name}")

    if not vectors:
        raise SystemExit("No usable face photos found.")
    mean = torch.stack(vectors).mean(dim=0)
    mean = torch.nn.functional.normalize(mean, dim=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean, args.output)
    print(f"Saved owner embedding from {len(vectors)} photos to {args.output}")


if __name__ == "__main__":
    main()
