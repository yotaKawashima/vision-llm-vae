from pathlib import Path
import pandas as pd
from pycocotools.coco import COCO


class NSDCaptionReader:
    """
    Class for reading COCO captions for the NSD dataset.
    """

    def __init__(self, nsd_stimulus_info_path, coco_caption_dir):
        self.coco_caption_dir = coco_caption_dir
        stimulus_info_file = Path(nsd_stimulus_info_path)
        self.stimulus_info = pd.read_pickle(stimulus_info_file)
        self.num_stimuli = len(self.stimulus_info)

        coco_caption_file_train = (
            Path(self.coco_caption_dir) / "captions_train2017.json"
        )
        coco_caption_file_val = Path(self.coco_caption_dir) / "captions_val2017.json"

        self.coco_caption_train = COCO(coco_caption_file_train)
        self.coco_caption_val = COCO(coco_caption_file_val)

    def extract_coco_captions(self, index):
        """Read five captions for a given table index.

        Parameters
        ----------
        index : int
            Index of the table row for which to extract captions

        Returns
        -------
        tuple (list, int)
            List of captions for the image and the cocoId for the image.
        """

        coco_captions = []

        row = self.stimulus_info.iloc[index]
        if row["cocoSplit"] == "train2017":
            coco_caption_idx = self.coco_caption_train.getAnnIds(imgIds=[row["cocoId"]])
            coco_caption = self.coco_caption_train.loadAnns(coco_caption_idx)
            coco_captions = [caption_obj["caption"] for caption_obj in coco_caption]
        elif row["cocoSplit"] == "val2017":
            coco_caption_idx = self.coco_caption_val.getAnnIds(imgIds=[row["cocoId"]])
            coco_caption = self.coco_caption_val.loadAnns(coco_caption_idx)
            coco_captions = [caption_obj["caption"] for caption_obj in coco_caption]
        return coco_captions, row["cocoId"]


if __name__ == "__main__":

    import sys
    import os
    import torch
    import json
    from tqdm import tqdm
    import torch.nn.functional as F

    # Add parent directory to path
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )

    import config

    model = config.get_model()

    # empty text embeddings
    empty_text = [""]
    empty_embedding = model.encode(empty_text, convert_to_tensor=True)
    # ensure directory exists using pathlib
    config.text_embeddings_dir_path.mkdir(parents=True, exist_ok=True)
    torch.save(empty_embedding, config.empty_embedding_data_path)

    # Use Coco dataset
    embedding_data_path = config.text_embeddings_nsd_path
    embedding_var_data_path = config.text_embeddings_nsd_var_path
    meta_data_path = config.text_embeddings_nsd_meta_path

    nsd_caption_reader = NSDCaptionReader(
        config.nsd_stimulus_info_path, config.coco_caption_dir
    )

    # get all captions
    embeddings = []
    embeddings_vars = []
    meta = []

    for row_id in tqdm(range(nsd_caption_reader.num_stimuli)):
        # extract captions
        texts, img_id = nsd_caption_reader.extract_coco_captions(row_id)
        embs = model.encode(texts, convert_to_tensor=True)
        if config.text_embeddings_summary:  # summarize representations
            mean_emb = torch.mean(embs, dim=0)
            # normalize mean embedding after taking mean
            mean_emb = F.normalize(mean_emb, p=2, dim=0)
            var_emb = torch.var(embs, dim=0)
            embeddings.append(mean_emb)
            embeddings_vars.append(var_emb)
            meta.append({"image_id": int(img_id), "captions": texts})
        else:
            embeddings.append(embs)
            for text in texts:
                meta.append({"image_id": int(img_id), "captions": [text]})

    embeddings = torch.stack(embeddings)
    if config.text_embeddings_summary:
        embeddings_vars = torch.stack(embeddings_vars)

    torch.save(embeddings, embedding_data_path)
    torch.save(embeddings_vars, embedding_var_data_path)

    with open(meta_data_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
