from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_split(splits_csv, split):
    df = pd.read_csv(splits_csv)
    return df[df["split"] == split].reset_index(drop=True)


def build_transforms(cfg, train):
    size = cfg["image_size"]
    if not train:
        return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    a = cfg["augment"]
    kernel = int(6 * a["blur_sigma_max"]) | 1  # odd kernel wide enough for the largest blur
    return T.Compose([
        T.RandomResizedCrop(size, scale=(a["scale_min"], 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(a["hflip"]),
        T.RandomRotation(a["rotation"]),
        T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
        # random blur so sharpness alone can't separate the grades
        T.RandomApply([T.GaussianBlur(kernel, sigma=(0.1, a["blur_sigma_max"]))], p=a["blur_p"]),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class FundusDataset(Dataset):
    def __init__(self, frame, image_dir, transform):
        self.names = frame["name"].tolist()
        self.labels = frame["grade"].astype(int).tolist()
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        with Image.open(self.image_dir / f"{self.names[i]}.jpg") as im:
            x = self.transform(im.convert("RGB"))
        return x, self.labels[i], self.names[i]
