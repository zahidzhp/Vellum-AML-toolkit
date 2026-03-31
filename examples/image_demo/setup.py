"""Generate a tiny synthetic image classification dataset (colored squares)."""

from pathlib import Path

import numpy as np
from PIL import Image

rng = np.random.RandomState(42)
base = Path(__file__).parent / "dataset"

colors = {
    "red": [200, 50, 50],
    "green": [50, 200, 50],
    "blue": [50, 50, 200],
}

for class_name, rgb in colors.items():
    class_dir = base / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        noise = rng.randint(-30, 30, (64, 64, 3))
        pixels = np.clip(np.array(rgb) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(class_dir / f"{class_name}_{i:02d}.jpg")

total = sum(1 for _ in base.rglob("*.jpg"))
print(f"Created {total} images in {base} ({len(colors)} classes, {total // len(colors)} per class)")
