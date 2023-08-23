import os
import numpy as np
from pathlib import Path
import cv2

dataset_dir = Path("/export/home/huang/multiview_processed_15hz/turn_on_the_green_light_processed_15hz")
target_dir = Path("/tmp")


def read_imgs(path: Path):
    sample = dict(np.load(path, allow_pickle=True))
    images = []
    for key in sorted(sample.keys()):
        if "rgb" in key:
            if len(sample[key].shape) < 3:
                continue
            images.append(sample[key][:, :, ::-1])

    return images


def main():
    imgs_dir = target_dir / (dataset_dir.parts[-1] + "_imgs")
    os.makedirs(imgs_dir, exist_ok=True)
    paths_list = sorted([i for i in dataset_dir.iterdir() if i.name.startswith("episode")])
    for path in paths_list:
        target_path = imgs_dir / (path.stem + ".png")
        imgs = read_imgs(path)
        imgs = np.concatenate(imgs, axis=1)
        cv2.imwrite(target_path.as_posix(), imgs)


if __name__ == "__main__":
    main()
