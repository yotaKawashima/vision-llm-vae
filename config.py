import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import numpy as np
import torchvision.transforms as T
from experiments.utils import replace_subdir  # get_last_checkpoint_path


############################################################################
###### text embeddings ######
text_embeddings_summary = True  # if true, use mean and var embeddings across captions
sentence_transformer_model_name = "all-mpnet-base-v2"
text_embedding_dim = 768
############################################################################

num_workers = 2
run_id = 0

############################################################################
####### Training Hyperparameters ######
number_of_epochs = 50
batch_size = 128
learning_rate = 0.0005
clip_grad_norm = 1.0
###### Model ######
model_type = "encoder"  # "encoder", "ae", "beta_vae", "beta_vae_llm"
resnet_flag = False
latent_dim = text_embedding_dim
checkpoint_path = None
# checkpoint_path = Path(
#     "/mnt/data/checkpoints/ae_loss_l2/cocoDoerig/run_0/checkpoint_epoch25.ckpt"
# )  # or None
encoder_checkpoint = False  # whether to initialize the encoder with the checkpoint from the encoder model (only applicable for ae model)
ae_checkpoint = False  # whether to initialize the ae model with the checkpoint from the ae model (only applicable for beta_vae model)
vae_checkpoint = False  # whether to initialize the vae model with the checkpoint from the vae model (only applicable for beta_vae_llm model)
last_checkpoint_path = None  # for activation extraction
loss_type = None
beta = None
gamma = None
recon_loss_type = None
llm_alignment_loss_type = None
alpha = None


MODEL_CONFIGS = {
    "encoder": {
        "loss_type": "norm_and_cosine_similarity_smoothL1",  # default loss
        "alpha": 10.0,  # weight for cosine similarity loss when loss_type is "norm_and_cosine_similarity_smoothL1" or "norm_and_cosine_similarity"
    },
    "ae": {
        "loss_type": "l2",
    },
    "beta_vae": {
        "loss_type": "llm_alignment",  # or "standard"
        "beta": 0.001,  # beta for KL divergence loss
        "recon_loss_type": "l2",
        "llm_alignment_loss_type": "cosine_similarity",
        "gamma": 1.0,  # weight for llm alignment loss when loss_type is "llm_alignment"
    },
    "beta_vae_llm": {
        "loss_type": "l2",
    },
}

# model to use
current_cfg = MODEL_CONFIGS[model_type]

# unpack the current config to global variables
for key, value in current_cfg.items():
    globals()[key] = value

if resnet_flag:
    full_model_name = "resnet18_" + model_type + f"_loss_{current_cfg['loss_type']}"
else:
    full_model_name = model_type + f"_loss_{current_cfg['loss_type']}"

if model_type == "encoder":
    full_model_name = full_model_name + f"_alpha{current_cfg['alpha']}"

elif model_type == "beta_vae":
    full_model_name = full_model_name + f"_beta{current_cfg['beta']}"

    full_model_name = full_model_name + f"_recon_loss_{current_cfg['recon_loss_type']}"

    if current_cfg["loss_type"] == "llm_alignment":
        full_model_name = (
            full_model_name
            + f"_llm_alignment_loss_{current_cfg['llm_alignment_loss_type']}"
            + f"_gamma{current_cfg['gamma']}"
        )
    else:
        assert (
            not ae_checkpoint
        ), "ae_checkpoint can only be True when loss_type is 'llm_alignment' for beta_vae model."
        full_model_name = "vanilla_" + full_model_name

elif model_type == "ae":
    if not encoder_checkpoint:
        full_model_name = "vanilla_" + full_model_name

elif model_type in ["beta_vae_llm"]:
    pass
else:
    raise ValueError(
        f"Unknown model type '{model_type}' specified in the configuration."
    )
############################################################################
###### Image Transform ######
if resnet_flag:
    img_resize = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
else:
    img_resize = 128
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]

if model_type in ["encoder"]:
    # PIL image
    img_transform_val = T.Compose(
        [
            T.Resize((img_resize, img_resize)),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )
    img_transform_train = T.Compose(
        [
            T.Resize((img_resize, img_resize)),
            T.RandomAffine(degrees=2, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )
    # h5: array of uint8 in HWC format, need to be converted to float CHW and normalized.
    img_transform_val_h5 = T.Compose(
        [
            T.ToTensor(),  # convert uint8 HWC [0, 255] to float CHW [0.0, 1.0]
            T.Resize((img_resize, img_resize)),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )
    img_transform_train_h5 = T.Compose(
        [
            T.ToTensor(),  # convert uint8 HWC [0, 255] to float CHW [0.0, 1.0]
            T.Resize((img_resize, img_resize)),
            T.RandomAffine(degrees=2, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )

else:  # no data augmentation for image reconstruction models
    # PIL image
    img_transform_val = T.Compose(
        [
            T.Resize((img_resize, img_resize)),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )
    img_transform_train = img_transform_val

    # h5
    img_transform_val_h5 = T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_resize, img_resize)),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )
    img_transform_train_h5 = img_transform_val_h5

############################################################################

############################################################################
###### Variable related to data path ######
data_dir_path = Path("/mnt/data/")
coco_version = "Doerig"  # "2017"
############################################################################

############################################################################
##### activation extraction for encoding model analysis #####
target_layers = [
    "ReLU",
]  # ["conv2"]
vision_bias = 0.5
input_modality = "image"
############################################################################

############################################################################
##### encoding model #####
# regression hyperparameters
# alphas = np.logspace(-3, 3, 7)
alphas = np.logspace(-6, 6, 13)
num_folds = 5

# list of roi (streams and all)
roi_list = [
    "early",
    "midventral",
    "ventral",
    "midlateral",
    "lateral",
    "midparietal",
    "parietal",
    "all",
]

subjects = [1, 2, 3, 4, 5, 6, 7, 8]
############################################################################


############################################################################
###### The followings are fixed paths and related functions ######
############################################################################
###### Paths ######
if coco_version == "Doerig":
    coco_doerig_h5_path = (
        data_dir_path / "ms_coco_embeddings_square256_proper_chunks.h5"
    )
else:
    raw_coco_data_dir_path = data_dir_path / "raw" / f"coco{coco_version}"
    coco_caption_train_path = (
        raw_coco_data_dir_path / "annotations" / f"captions_train{coco_version}.json"
    )
    coco_caption_val_path = (
        raw_coco_data_dir_path / "annotations" / f"captions_val{coco_version}.json"
    )
    coco_image_train_dir_path = (
        raw_coco_data_dir_path / "images" / f"train{coco_version}"
    )
    coco_image_val_dir_path = raw_coco_data_dir_path / "images" / f"val{coco_version}"


# preprocessed data
preprocessed_data_dir_path = data_dir_path / "preprocessed"

# test embeddings
text_embeddings_dir_path = preprocessed_data_dir_path / "text_embeddings"

if text_embeddings_summary:
    text_embeddings_file_name = "text_embeddings_mean"
    text_embeddings_var_file_name = "text_embeddings_var"
    text_embeddings_meta_file_name = "meta_text_embeddings_mean"
else:
    text_embeddings_file_name = "text_embeddings"
    text_embeddings_meta_file_name = "meta_text_embeddings"

text_embeddings_train_path = (
    text_embeddings_dir_path / f"{text_embeddings_file_name}_train{coco_version}.pt"
)
text_embeddings_val_path = (
    text_embeddings_dir_path / f"{text_embeddings_file_name}_val{coco_version}.pt"
)
if text_embeddings_summary:
    text_embeddings_train_var_path = (
        text_embeddings_dir_path
        / f"{text_embeddings_var_file_name}_train{coco_version}.pt"
    )
    text_embeddings_val_var_path = (
        text_embeddings_dir_path
        / f"{text_embeddings_var_file_name}_val{coco_version}.pt"
    )

text_embeddings_meta_train_path = (
    text_embeddings_dir_path
    / f"{text_embeddings_meta_file_name}_train{coco_version}.json"
)
text_embeddings_meta_val_path = (
    text_embeddings_dir_path
    / f"{text_embeddings_meta_file_name}_val{coco_version}.json"
)

empty_embedding_data_path = text_embeddings_dir_path / "empty_text_embedding.pt"

# training checkpoints
all_checkpoints_dir_path = data_dir_path / "checkpoints"
coco_checkpoints_dir_path = (
    all_checkpoints_dir_path / full_model_name / f"coco{coco_version}" / f"run_{run_id}"
)
coco_checkpoints_dir_path.mkdir(parents=True, exist_ok=True)

# writer path for tensorboard logs
writer_path = (
    data_dir_path
    / "tensorboard_runs"
    / full_model_name
    / f"coco{coco_version}"
    / f"run_{run_id}"
)
training_history_path = coco_checkpoints_dir_path / "training_history.json"

# model activations data
if last_checkpoint_path is not None:
    model_activation_file_name = last_checkpoint_path.parent / (
        last_checkpoint_path.stem + "_activations.pt"
    )
    model_activation_path = replace_subdir(
        model_activation_file_name,
        "checkpoints",
        "model_activations",
    )

    model_activation_path = model_activation_path / input_modality

# fmri data
nsd_stimulus_info_path = data_dir_path / "nsd" / "nsd_stimulus_info.pkl"

fmri_dir_name = "nsd_fmri"
fmri_file_name = "betas_average_fsaverage"
fmri_file_extension = ".npy"
fmri_dir_path = data_dir_path / fmri_dir_name
# averaged_fmri_data_path = fmri_dir_path / f"averaged_{fmri_file_name}_test{fmri_file_extension}"

# encoding model data
encoding_model_dir_name = "encoding_model"
encoding_model_file_name = "encoding_model"
encoding_model_file_extension = ".npz"
encoding_model_dir_path = data_dir_path / encoding_model_dir_name

###### Functions ######
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(sentence_transformer_model_name)
    return _model
