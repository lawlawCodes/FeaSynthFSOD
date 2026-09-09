"""
extract_Word2Vec_feature.py
============================
Extract Word2Vec-based semantic embeddings for object categories,
mirroring the CLIP version's interface (extract_CLIP_feature.py).

Differences from CLIP version:
  - Embedding dim: 300 (Word2Vec) vs 512 (CLIP)
  - Text encoding: word-level averaging via gensim instead of CLIP tokenizer+encoder
  - Save path: side_information/Word2Vec/ instead of side_information/CLIP/

Usage:
  python extract_Word2Vec_feature.py --w2v_model /path/to/GoogleNews-vectors-negative300.bin
"""

import pdb
import torch
import torch.nn.functional as F
import numpy as np
import re
import os

from defrcn.data.builtin_meta import (
    COCO_CATEGORIES,
    COCO_NOVEL_CATEGORIES,
    PASCAL_VOC_ALL_CATEGORIES,
    PASCAL_VOC_NOVEL_CATEGORIES,
    PASCAL_VOC_BASE_CATEGORIES,
)

import ast
import argparse

# --------------------- paths ---------------------
ROOT_PATH = "side_information/Word2Vec/"

parser = argparse.ArgumentParser(description="Extract Word2Vec class embeddings")
parser.add_argument(
    "--w2v_model",
    default="/home/dragon/GoogleNews-vectors-negative300.bin",
    help="Path to pretrained Word2Vec binary model (GoogleNews format)",
)
parser.add_argument("--device", default="cuda:0", help="cpu / cuda:x")


# --------------------- text preprocessing ---------------------
# Stop words to filter out when extracting key words from descriptions
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "has", "have", "with", "of", "in",
    "on", "to", "for", "and", "or", "its", "it", "that", "this", "be",
    "as", "by", "at", "from", "can", "may", "often", "usually", "there",
    "several", "one", "two", "three", "more", "such",
}


def _tokenize_words(text):
    """Lower-case, split on non-alpha, filter stop words and short tokens."""
    tokens = re.findall(r"[a-z]{2,}", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


# --------------------- core functions ---------------------
def extract_class_embeddings(dataset, model, device, category_descriptions):
    """
    Parameters
    ----------
    dataset : str
        Dataset identifier, e.g. "voc_base1", "coco_all".
    model : gensim KeyedVectors
        Loaded Word2Vec model.
    device : str
        Target device for output tensor.
    category_descriptions : dict[str, str]
        Mapping from class name to textual description.

    Returns
    -------
    classes : list[str]
        Ordered category names.
    class_embeddings : torch.Tensor  [num_classes, 300]
        L2-normalized Word2Vec embeddings.
    """
    # ------ resolve class list ------
    classes = []

    if "voc" in dataset:
        split = int(dataset.split("_")[-1][-1])
        if "base" in dataset:
            classes = PASCAL_VOC_BASE_CATEGORIES[split]
        elif "novel" in dataset:
            classes = PASCAL_VOC_NOVEL_CATEGORIES[split]
        elif "all" in dataset:
            classes = PASCAL_VOC_ALL_CATEGORIES[split]

    if "coco" in dataset:
        all_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        novel_classes = [k["name"] for k in COCO_NOVEL_CATEGORIES if k["isthing"] == 1]
        base_classes = [
            k["name"]
            for k in COCO_CATEGORIES
            if k["isthing"] == 1 and k["name"] not in novel_classes
        ]
        if "base" in dataset:
            classes = base_classes
        elif "novel" in dataset:
            classes = novel_classes
        elif "all" in dataset:
            classes = all_classes

    # ------ extract embeddings ------
    category_text_features = []
    oov_count = 0  # out-of-vocabulary word counter

    for cls_name in classes:
        desc = category_descriptions.get(cls_name, cls_name)

        # Tokenize the description into meaningful words
        words = _tokenize_words(desc)

        # Collect word vectors (skip OOV words)
        word_vecs = []
        for w in words:
            if w in model:
                word_vecs.append(torch.from_numpy(model[w]).float())
            else:
                oov_count += 1

        if len(word_vecs) == 0:
            # Fallback: if ALL words are OOV, use zero vector
            print(f"  [WARN] All words OOV for '{cls_name}', using zero vector")
            cls_emb = torch.zeros(model.vector_size)
        else:
            # Average word vectors → sentence-level embedding
            cls_emb = torch.stack(word_vecs, dim=0).mean(dim=0)

        # L2 normalize (same post-processing as CLIP version)
        cls_emb = F.normalize(cls_emb, dim=-1, p=2)

        category_text_features.append(cls_emb)

    if oov_count > 0:
        print(f"  [INFO] {oov_count} out-of-vocabulary word(s) skipped")

    class_embeddings = torch.stack(category_text_features, dim=0).to(device)
    return classes, class_embeddings


def load_class_embeddings(dataset, device):
    """
    Load precomputed Word2Vec embeddings from .npy files.
    Mirrors the CLIP version's interface for drop-in compatibility.
    """
    if "voc" in dataset:
        split = int(re.search(r"(?:base|all)(\d+)", dataset).group(1))

        if "base" in dataset:
            loadpath = f"{ROOT_PATH}voc_base{split}_class_embeddings.npy"
        elif "novel" in dataset:
            loadpath = f"{ROOT_PATH}voc_novel{split}_class_embeddings.npy"
        elif "all" in dataset:
            loadpath = f"{ROOT_PATH}voc_all{split}_class_embeddings.npy"

    if "coco" in dataset:
        if "base" in dataset:
            loadpath = f"{ROOT_PATH}coco_base_class_embeddings.npy"
        elif "novel" in dataset:
            loadpath = f"{ROOT_PATH}coco_novel_class_embeddings.npy"
        elif "all" in dataset:
            loadpath = f"{ROOT_PATH}coco_all_class_embeddings.npy"

    return torch.from_numpy(np.load(loadpath)).to(device)


# --------------------- main ---------------------
def main():
    from gensim.models import KeyedVectors

    args = parser.parse_args()
    device = args.device

    # ------ load Word2Vec model ------
    print(f"Loading Word2Vec model from: {args.w2v_model}")
    model = KeyedVectors.load_word2vec_format(args.w2v_model, binary=True)
    print(f"  Loaded. vocab_size={len(model)}, dim={model.vector_size}")

    # ------ ensure output directory ------
    os.makedirs(ROOT_PATH, exist_ok=True)

    # ------ load category descriptions ------
    desc_dir = "side_information"

    # VOC descriptions
    with open(f"{desc_dir}/voc_descriptions_v3.txt", "r", encoding="utf-8") as f:
        voc_descriptions = ast.literal_eval(f.read())

    # COCO descriptions
    with open(f"{desc_dir}/coco_descriptions_v2.txt", "r", encoding="utf-8") as f:
        coco_descriptions = ast.literal_eval(f.read())

    # ------ datasets to process ------
    dataset_list = [
        "voc_base1", "voc_base2", "voc_base3",
        "voc_novel1", "voc_novel2", "voc_novel3",
        "voc_all1", "voc_all2", "voc_all3",
        "coco_base", "coco_novel", "coco_all",
    ]

    for dataset in dataset_list:
        print(f"\nProcessing: {dataset}")

        # pick the right description dict
        if "voc" in dataset:
            descriptions = voc_descriptions
        else:
            descriptions = coco_descriptions

        classnames, class_embeddings = extract_class_embeddings(
            dataset, model, device, descriptions
        )

        save_path = f"{ROOT_PATH}{dataset}_class_embeddings.npy"
        np.save(save_path, class_embeddings.cpu().numpy())
        print(f"  Saved → {save_path}  shape={class_embeddings.shape}")

    print("\nDone. All embeddings saved to", ROOT_PATH)


if __name__ == "__main__":
    main()
