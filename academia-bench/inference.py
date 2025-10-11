import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models
from detectors import DETECTOR
from transform import get_albumentations_transforms, get_albumentations_transforms_vit_clip
from models.clip import clip 
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class CSVImageDataset(Dataset):
    """
    Expects a CSV with at least two columns:
      - image path column (default: 'Image Path')
      - label column (default: 'label')
    You can override names via --image_col and --label_col.
    """
    def __init__(self, df, image_col: str, label_col: str, owntransforms):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = owntransforms

        # Optional: filter out rows whose files don't exist
        missing = []
        keep_rows = []
        for i, row in self.df.iterrows():
            p = str(row[self.image_col])
            if isinstance(p, str) and os.path.isfile(p):
                keep_rows.append(True)
            else:
                keep_rows.append(False)
                missing.append(p)
        if missing:
            print(f"[WARN] Skipping {len(missing)} missing files. Example: {missing[:3]}")
        self.df = self.df[pd.Series(keep_rows).values].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = str(row[self.image_col])
        label = int(row[self.label_col])

        img = np.array(Image.open(image_path).convert("RGB"))
        augmented = self.transform(image=img)
        img = augmented["image"]

        # Model seems to expect dict with 'image' and 'label'
        return {
            "image": img,
            "label": label,
            "image_path": image_path,
        }

class CLIPModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load('ViT-L/14', device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( 768, num_classes )
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str,default='../dataset/image.csv', help="Input CSV with image paths and labels")
    parser.add_argument("--output_csv", type=str, default='../results/post_processing/image/image_UnivFD.csv', help="Where to save predictions CSV")
    parser.add_argument("--image_col", type=str, default="Image Path", help="Column name for image paths")
    parser.add_argument("--label_col", type=str, default="Target", help="Column name for labels (0/1)")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/UnivFD.pth")
    parser.add_argument("--model_structure", type=str, default="UnivFD", help="Detector key in DETECTOR")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Transforms
    # === Load appropriate transform ===
    if args.model_structure == 'vit':
        test_transforms = get_albumentations_transforms_vit_clip([''], model_type='vit')
    elif args.model_structure == 'UnivFD':
        test_transforms = get_albumentations_transforms_vit_clip(['gaussian_blur'], model_type='clip')
    else:
        test_transforms = get_albumentations_transforms([''])


    # Model
    # model_class = DETECTOR[args.model_structure]
    # model = model_class()
    # if args.model_structure == 'vit':
    #     model = models.vit_b_16(pretrained=True)
    #     model.heads[0] = nn.Linear(768, 1)
    model = CLIPModel()
    torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)
    params = []
    for name, p in model.named_parameters():
        if  name=="fc.weight" or name=="fc.bias": 
            params.append(p) 
        else:
            p.requires_grad = False
    model.load_state_dict(torch.load(args.checkpoints, map_location=device, weights_only=True), strict=True)
    model.to(device).eval()
    print(f"Loaded model: {args.model_structure}")

    # Data
    df = pd.read_csv(args.csv_path)
    if args.image_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.image_col}' and '{args.label_col}'. "
                         f"Found: {list(df.columns)}")
    # print(test_transforms)
    dataset = CSVImageDataset(df, args.image_col, args.label_col, test_transforms)
    if len(dataset) == 0:
        raise RuntimeError("No valid images to process (after filtering missing files).")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Inference loop
    all_paths = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing", ncols=80):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]  # keep on CPU for saving
            # Model API expects a dict with 'image' (and often 'label'), plus inference=True
            # output = model({"image": images, "label": labels}, inference=True)

            # # Handle shape: output['cls'] could be [B], [B,1], or [B,2] (logits)
            # logits = output["cls"]
            # if logits.ndim == 2 and logits.size(1) == 1:
            #     logits = logits.squeeze(1)
            # elif logits.ndim == 2 and logits.size(1) == 2:
            #     # assume binary logits [real, fake] -> take fake logit
            #     logits = logits[:, 1]
            # elif logits.ndim != 1:
            #     raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

            output = model(images)
            logits = output

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()

            all_paths.extend(batch["image_path"])
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    out_df = pd.DataFrame({
        "image_path": all_paths,
        "label": all_labels,
        "prob_fake": all_probs,
    })
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
