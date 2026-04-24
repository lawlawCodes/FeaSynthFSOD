import pdb
import torch
import clip
import torch.nn.functional as F
import numpy as np

from FeaSynthFSOD.data.builtin_meta import (
    COCO_CATEGORIES,
    COCO_NOVEL_CATEGORIES,
    PASCAL_VOC_ALL_CATEGORIES,
    PASCAL_VOC_NOVEL_CATEGORIES,
    PASCAL_VOC_BASE_CATEGORIES,
)

import ast
import argparse

rootPath = "text_embeddings/CLIP/"

parser = argparse.ArgumentParser(description="")
parser.add_argument('--backbone', default='ViT-B/16', help='CLIP backbone')
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')


def extract_text_embeddings(dataset, model, device):

    classes = []

    if 'voc' in dataset:
        with open(f"text_descriptions/voc.txt", "r", encoding="utf-8") as f:
            txt_content = f.read()
            category_descriptions = ast.literal_eval(txt_content)

        split = int(dataset.split('_')[-1][-1])
        if 'base' in dataset:
            classes = PASCAL_VOC_BASE_CATEGORIES[split]
        if 'novel' in dataset:
            classes = PASCAL_VOC_NOVEL_CATEGORIES[split]
        if 'all' in dataset:
            classes = PASCAL_VOC_ALL_CATEGORIES[split]
        dataset = ""

    if 'coco' in dataset:
        with open(f"text_descriptions/coco.txt", "r", encoding="utf-8") as f:
            txt_content = f.read()
            category_descriptions = ast.literal_eval(txt_content)
        all_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        novel_classes = [k["name"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
        base_classes = [k['name'] for k in COCO_CATEGORIES if k["isthing"] == 1 and k["name"] not in novel_classes]
        if 'base' in dataset:
            classes = base_classes
        if 'novel' in dataset:
            classes = novel_classes
        if 'all' in dataset:
            classes = all_classes
        dataset = ""

    category_text_features = []

    with torch.no_grad():
        for cls_name in classes:
            prompts = clip.tokenize(category_descriptions[cls_name]).cuda()
            with torch.no_grad():
                text_features = model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1, p=2).float().mean(dim=0)

                category_text_features.append(text_features.cpu())

    return classes, torch.stack(category_text_features, dim=0).to(device)


def load_text_embeddings(dataset, device):

    if 'voc' in dataset:
        split = int(__import__("re").search(r"(?:base|all)(\d+)", dataset).group(1))

        if 'base' in dataset:
            loadpath = f"{rootPath}voc_base{split}_text_embeddings.npy"
        if 'novel' in dataset:
            loadpath = f"{rootPath}voc_novel{split}_text_embeddings.npy"
        if 'all' in dataset:
            loadpath = f"{rootPath}voc_all{split}_text_embeddings.npy"

        dataset = ""

    if 'coco' in dataset:
        if 'base' in dataset:
            loadpath = f"{rootPath}coco_base_text_embeddings.npy"
        if 'novel' in dataset:
            loadpath = f"{rootPath}coco_novel_text_embeddings.npy"
        if 'all' in dataset:
            loadpath = f"{rootPath}coco_all_text_embeddings.npy"

        dataset = ""

    return torch.from_numpy(np.load(loadpath)).to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()

    model, preprocess = clip.load(args.backbone, device=args.device)
    model.eval()

    dataset_dict = [
                    "voc_all1",
                    "voc_all2",
                    "voc_all3",
                    "voc_base1",
                    "voc_base2",
                    "voc_base3",
                    "coco_base",
                    "coco_all"
                    ]

    for dataset in dataset_dict:

        classnames, class_embeddings = extract_text_embeddings(dataset, model, device)

        save_path = f"{rootPath}{dataset}_text_embeddings.npy"

        np.save(save_path, class_embeddings.cpu().numpy())


if __name__ == "__main__":
    main()
