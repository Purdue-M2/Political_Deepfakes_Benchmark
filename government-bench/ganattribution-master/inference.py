import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import dct
import freq_res
import albumentations as A
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def is_image_file(filename):
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        # Handle both old/new torchvision weight APIs
        try:
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = models.resnet101(pretrained=True)

        self.features = nn.Sequential(*list(backbone.children())[:-1])  # global pool output
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        out = self.features(x)                    # [B, 2048, 1, 1]
        out = out.reshape(out.size(0), -1)        # [B, 2048]
        logits = self.fc(out)                     # [B, 2]
        probs = F.softmax(logits, dim=1)          # class-0=real, class-1=fake
        return probs
    

# ---------------------------
# Albumentations builder 
# ---------------------------
def get_albumentations(use_methods):
    transforms_list = []

    if 'gaussian_blur' in use_methods:
        transforms_list.append(A.GaussianBlur(blur_limit=(5, 5), p=1.0))
        print('gaussian_blur kernel size 5')
    if 'jpeg_compression' in use_methods:
        transforms_list.append(A.ImageCompression(quality_lower=80, quality_upper=80, p=1.0))
        print('using jpeg compression 80')
    if 'random_crop' in use_methods:
        transforms_list.append(A.RandomCrop(height=224, width=224, p=1.0))
        print('random_crop_224')
    if 'center_crop' in use_methods:
        transforms_list.append(A.CenterCrop(height=224, width=224, p=1.0))
        print('center_crop_224')
    if 'hue_saturation_value' in use_methods:
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=1.0))
        print('HueSaturationValue applied')
    if 'random_brightness_contrast' in use_methods:
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0))
        print('BrightnessContrast applied')
    if 'rotation' in use_methods:
        transforms_list.append(A.Rotate(limit=30, p=1.0))
        print('rotation applied with limit 30')

    # If empty, Compose([]) is fine (no-op)
    return A.Compose(transforms_list)

# Wrapper: PIL -> Albumentations -> PIL
def albumentations_to_pil(albumentations_transform):
    def wrapper(img: Image.Image) -> Image.Image:
        arr = np.array(img)  # PIL -> np (H,W,C, uint8)
        out = albumentations_transform(image=arr)["image"]
        return Image.fromarray(out)  # np -> PIL
    return wrapper

class CSVImageDataset(Dataset):
    def __init__(self, csv_path, image_col="image_path", transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_col = image_col
        if self.image_col not in self.df.columns:
            raise ValueError(f"Column '{self.image_col}' not found in CSV.")
        self.paths = self.df[self.image_col].astype(str).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            if not (os.path.isfile(path) and is_image_file(path)):
                return path, None
            img = default_loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return path, img
        except Exception:
            return path, None

def evaluate_csv(
    csv_path,
    output_csv_path,
    model_load_path="models/model_100.pth",
    image_col="image_path",
    num_workers=1,
    batch_size=16,
    use_methods=None, 
    use_cuda=True
):
    
    if use_methods is None:
        use_methods = []

    st = time.time()

    if not os.path.exists(model_load_path):
        raise Exception(f"Model path {model_load_path} not found")

    resize_dims = 256
    # transform_img = transforms.Compose([
    #     transforms.Resize(resize_dims),
    #     transforms.CenterCrop(resize_dims),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                          std=[0.5, 0.5, 0.5])
    # ])
    # Albumentations first (acts on PIL via wrapper)
    a_transform = get_albumentations(use_methods)
    tv_steps = []
    tv_steps.append(transforms.Lambda(albumentations_to_pil(a_transform)))

    # Your existing geometric pipeline
    tv_steps.extend([
        transforms.Resize(resize_dims),
        transforms.CenterCrop(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    transform_img = transforms.Compose(tv_steps)

    print(transform_img)

    device = torch.device("cpu") if not use_cuda else torch.device("cuda:0")
    checkpoint = torch.load(model_load_path, map_location=device)
    model = nn.DataParallel(BinaryNet()).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv(csv_path)
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in CSV.")
    ds = CSVImageDataset(csv_path, image_col=image_col, transform=transform_img)

    # Output columns: probability of fake and predicted label (argmax)
    if "PredictedLabel" not in df.columns:
        df["PredictedLabel"] = np.nan

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        collate_fn=lambda batch: list(zip(*batch))
    )

    path_to_indices = {}
    for i, p in enumerate(df[image_col].astype(str).tolist()):
        path_to_indices.setdefault(p, []).append(i)

    with torch.no_grad():
        for paths, imgs in loader:
            valid_idx = [i for i, im in enumerate(imgs) if im is not None]
            if not valid_idx:
                continue

            batch_tensor = torch.stack([imgs[i] for i in valid_idx], dim=0).to(device, non_blocking=True)
            probs = model(batch_tensor)                    # [B, 2], softmax
            pred = probs.argmax(dim=1)                     # labels: 0=real, 1=fake
            prob_fake = probs[:, 1]                        # probability of class "fake"

            for j, vi in enumerate(valid_idx):
                p = paths[vi]
                pred_label = int(pred[j].item())
                prob_val = float(prob_fake[j].item())
                for row_idx in path_to_indices.get(p, []):
                    df.at[row_idx, "PredictedLabel"] = pred_label
                    df.at[row_idx, "prob_fake"] = prob_val

    df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to: {output_csv_path}")
    print('Total time taken: {0:.2f} s'.format(time.time()-st))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BinaryNet predictions on images listed in a CSV")
    parser.add_argument("--input", default='/mnt/ssd/project/lilin/politic_deepfakes/AI-Face-FairnessBench-main/dataset/random_pick_10_crop.csv', help="Path to input CSV file")
    parser.add_argument("--output", default='/mnt/ssd/project/lilin/politic_deepfakes/AI-Face-FairnessBench-main/results/post_processing/video/random_pick_10_crop_random_brightness_contrast_ganattribution.csv', help="Path to output CSV file")
    parser.add_argument("--model", default="models/model_100.pth", help="Path to model checkpoint")
    parser.add_argument("--image_col", default="image_path", help="Column name in CSV with image paths")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of DataLoader workers")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    use_methods = [
        # "gaussian_blur",
        # "jpeg_compression",
        # "rotation",
        # "random_crop",
        # "center_crop",
        # "hue_saturation_value",
        "random_brightness_contrast",
    ]

    evaluate_csv(
        csv_path=args.input,
        output_csv_path=args.output,
        model_load_path=args.model,
        image_col=args.image_col,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        use_cuda=(not args.no_cuda and torch.cuda.is_available()),
        use_methods=use_methods,
    )
