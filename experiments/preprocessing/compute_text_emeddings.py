import sys
import os
import torch
import json
from tqdm import tqdm
import torch.nn.functional as F

# from sentence_transformers import SentenceTransformer
from pycocotools.coco import COCO

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import config

model = config.get_model()

# empty text embeddings
empty_text = [""]
empty_embedding = model.encode(empty_text, convert_to_tensor=True)
# ensure directory exists using pathlib
config.text_embeddings_dir_path.mkdir(parents=True, exist_ok=True)
torch.save(empty_embedding, config.empty_embedding_data_path)

# Use Coco dataset
for split in ["train", "val"]:
    if split == "train":
        caption_path = config.coco_caption_train_path
        embedding_data_path = config.text_embeddings_train_path
        embedding_var_data_path = config.text_embeddings_train_var_path
        meta_data_path = config.text_embeddings_meta_train_path
    else:
        caption_path = config.coco_caption_val_path
        embedding_data_path = config.text_embeddings_val_path
        embedding_var_data_path = config.text_embeddings_val_var_path
        meta_data_path = config.text_embeddings_meta_val_path

    # initialize COCO api
    coco_caps = COCO(caption_path)

    # get image ids
    image_ids = coco_caps.getImgIds()

    # get all captions
    embeddings = []
    embeddings_vars = []
    meta = []
    for img_id in tqdm(image_ids):
        # get annotation ids for the image
        ann_ids = coco_caps.getAnnIds(imgIds=img_id)

        # load captions
        captions = coco_caps.loadAnns(ann_ids)

        # extract captions
        texts = [caption_obj["caption"] for caption_obj in captions]
        embs = model.encode(texts, convert_to_tensor=True)
        if config.text_embeddings_summary:  # summarize representations
            mean_emb = torch.mean(embs, dim=0)
            # normalize mean embedding after taking mean
            mean_emb = F.normalize(mean_emb, p=2, dim=0)
            var_emb = torch.var(embs, dim=0)
            embeddings.append(mean_emb)
            embeddings_vars.append(var_emb)
            meta.append({"image_id": img_id, "captions": texts})
        else:
            embeddings.extend(embs)
            for text in texts:
                meta.append({"image_id": img_id, "captions": [text]})

    embeddings = torch.stack(embeddings)
    embeddings_vars = torch.stack(embeddings_vars)

    torch.save(embeddings, embedding_data_path)
    torch.save(embeddings_vars, embedding_var_data_path)

    with open(meta_data_path, "w") as f:
        json.dump(meta, f)
