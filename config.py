import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import numpy as np
import torchvision.transforms as T
from experiments.utils import replace_subdir

############################################################################
###### text embeddings ######
text_embeddings_summary = True  # if true, use mean and var embeddings across captions
sentence_transformer_model_name = "all-mpnet-base-v2"
text_embedding_dim = 768
pca_n_components = text_embedding_dim
############################################################################
num_workers = 0
run_id = 1

############################################################################
####### Training Hyperparameters ######
number_of_epochs = 50
batch_size = 128
learning_rate = 0.0005
clip_grad_norm = 1.0
###### Model ######
model_type = "encoder"  # "encoder", "ae", "beta_vae", "beta_vae_llm", "decoder"
resnet_flag = False
eval_flag = False  # true to make relevant dirs for evaluation and activation extraction
latent_dim = text_embedding_dim
checkpoint_path = None
# checkpoint_path = Path(
#     "/mnt/data/checkpoints/beta_vae_loss_llm_alignment_beta0.001_recon_loss_l2_llm_alignment_loss_cosine_similarity_gamma0.5/cocoDoerig/run_0/checkpoint_epoch30.ckpt"
# )
#     "/mnt/data/checkpoints/vanilla_ae_loss_l2/cocoDoerig/run_0/checkpoint_epoch50.ckpt"
# )  # or None
encoder_checkpoint = False  # whether to initialize the encoder with the checkpoint from the encoder model (only applicable for ae model)
ae_checkpoint = False  # whether to initialize the ae model with the checkpoint from the ae model (only applicable for beta_vae model)
vae_checkpoint = False  # whether to initialize the vae model with the checkpoint from the vae model (only applicable for beta_vae_llm model)
loss_type = None
beta = None
gamma = None
delta = None
recon_loss_type = None
llm_alignment_loss_type = None
alpha = None
temperature = None


MODEL_CONFIGS = {
    "encoder": {
        "loss_type": "norm_and_cosine_similarity_smoothL1",  # default loss
        "alpha": 10.0,  # weight for cosine similarity loss when loss_type is "norm_and_cosine_similarity_smoothL1" or "norm_and_cosine_similarity"
        "temperature": None,  # temperature for soft_nn alignment loss
    },
    "ae": {
        "loss_type": "l2",
    },
    "beta_vae": {
        "loss_type": "llm_alignment",  # or "llm_alignment"
        "beta": 0.001,  # beta for KL divergence loss
        "recon_loss_type": "l2",
        "llm_alignment_loss_type": "cosine_similarity",
        "gamma": 0.5,  # weight for llm alignment loss when loss_type is "llm_alignment"
        "temperature": None,  # temperature for soft_nn alignment loss
    },
    "beta_vae_llm": {
        "loss_type": "l2_and_img_norm",  # "l2" or "img_norm" (a custom loss that matches the norm of the latent representation with the norm of the text embeddings after scaling)
        "delta": 0.1,  # weight for the norm loss when loss_type is "l2_and_img_norm"
    },
    "decoder": {
        "loss_type": "l2",
    },
}
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
coco_version = "Doerig"
coco_caption_dir = data_dir_path / "coco"
nsd_stimulus_path = data_dir_path / "nsd" / "nsd_stim" / "nsd_stimuli.hdf5"
nsd_stimulus_info_dir_path = data_dir_path / "nsd" / "nsd_stim"
nsd_stimulus_info_path = nsd_stimulus_info_dir_path / "nsd_stim_info_merged.pkl"
roi_defs_dir_path = data_dir_path / "nsd" / "nsd_roi_defs"
fmri_dir_path = data_dir_path / "nsd" / "nsd_fmri"
fmri_rdm_dir_path = data_dir_path / "fmri_rdms"


def nsd_stimulus_info_path_this_subject(subject):
    if subject == "special515":
        return nsd_stimulus_info_dir_path / "nsd_stim_info_special515.pkl"
    else:
        return (
            nsd_stimulus_info_dir_path
            / f"subj{int(subject):02d}_nsd_stim_info_NOTspecial515.pkl"
        )


############################################################################

############################################################################
##### activation extraction for encoding model analysis #####
target_layers = [
    "encoder.1.2",
    "encoder.2.2",
    "encoder.4.2",
    "encoder.6.2",
    "encoder.7.2",
    "encoder.8.2",
    "encoder.9.2",
    "encoder.11.2",
    "encoder.13.2",
    "decoder.1.2",
    "decoder.3.2",
    "decoder.5.2",
    "decoder.6.2",
    "decoder.7.2",
    "decoder.8.2",
    "decoder.10.2",
    "decoder.12.2",
    "decoder.13.2",
    "latent",
    "mu",
]
# target_layers = ["ReLU"]
vision_bias = 0.5
input_modality = "image"
############################################################################

############################################################################
##### rsa #####
# regression hyperparameters
# alphas = np.logspace(-3, 3, 7)
alphas = np.logspace(-6, 6, 13)
num_folds = 5

# list of roi (streams and all)
roi_class = "streams"
list_rois = [
    "early",
    "midventral",
    "ventral",
    "midlateral",
    "lateral",
    "midparietal",
    "parietal",
    "all",
]

list_subjects = [1, 2, 3, 4, 5, 6, 7, 8]
############################################################################


############################################################################
###### The followings are fixed paths and related functions ######
############################################################################

# model to use
current_cfg = MODEL_CONFIGS[model_type]

if model_type == "encoder" and current_cfg.get("loss_type") in [
    "soft_nn",
    "norm_and_soft_nn",
]:
    if current_cfg.get("temperature") is None:
        raise ValueError(
            f"Temperature must be specified for encoder soft_nn-based loss. Current config: {current_cfg}"
        )

if (
    model_type == "beta_vae"
    and current_cfg.get("loss_type") == "llm_alignment"
    and current_cfg.get("llm_alignment_loss_type") == "soft_nn"
):
    if current_cfg.get("temperature") is None:
        raise ValueError(
            f"Temperature must be specified for beta_vae soft_nn alignment loss. Current config: {current_cfg}"
        )

if model_type == "beta_vae" and current_cfg.get("loss_type") == "llm_alignment":
    if current_cfg.get("llm_alignment_loss_type") not in [
        "cosine_similarity",
        "soft_nn",
    ]:
        raise ValueError(
            "Supported llm alignment loss types for BetaVAE: cosine_similarity, soft_nn."
        )
# unpack the current config to global variables
for key, value in current_cfg.items():
    globals()[key] = value

if resnet_flag:
    full_model_name = "resnet18_" + model_type + f"_loss_{current_cfg['loss_type']}"
else:
    full_model_name = model_type + f"_loss_{current_cfg['loss_type']}"

if model_type == "encoder":
    full_model_name = full_model_name + f"_alpha{current_cfg['alpha']}"

    if (
        current_cfg.get("loss_type") in ["soft_nn", "norm_and_soft_nn"]
        and current_cfg.get("temperature") is not None
    ):
        full_model_name = full_model_name + f"_temp{current_cfg['temperature']}"

elif model_type == "beta_vae":
    full_model_name = full_model_name + f"_beta{current_cfg['beta']}"

    full_model_name = full_model_name + f"_recon_loss_{current_cfg['recon_loss_type']}"

    if current_cfg["loss_type"] == "llm_alignment":
        full_model_name = (
            full_model_name
            + f"_llm_alignment_loss_{current_cfg['llm_alignment_loss_type']}"
            + f"_gamma{current_cfg['gamma']}"
        )
        if (
            current_cfg.get("llm_alignment_loss_type") == "soft_nn"
            and current_cfg.get("temperature") is not None
        ):
            full_model_name = full_model_name + f"_temp{current_cfg['temperature']}"
    else:
        if ae_checkpoint:
            full_model_name = "vanilla_from_ae_" + full_model_name
        else:
            full_model_name = "vanilla_" + full_model_name


elif model_type == "ae":
    if not encoder_checkpoint:
        full_model_name = "vanilla_" + full_model_name

elif model_type == "beta_vae_llm":
    full_model_name = full_model_name + f"_delta{current_cfg['delta']}"

elif model_type in ["decoder"]:
    pass
else:
    raise ValueError(
        f"Unknown model type '{model_type}' specified in the configuration."
    )


###### Paths ######
if coco_version == "Doerig":
    coco_doerig_h5_path = (
        coco_caption_dir / "ms_coco_embeddings_square256_proper_chunks.h5"
    )

else:
    raise ValueError(
        f"Unsupported COCO version '{coco_version}' specified in the configuration."
    )
    # raw_coco_data_dir_path = data_dir_path / "raw" / f"coco{coco_version}"
    # coco_caption_train_path = (
    #     raw_coco_data_dir_path / "annotations" / f"captions_train{coco_version}.json"
    # )
    # coco_caption_val_path = (
    #     raw_coco_data_dir_path / "annotations" / f"captions_val{coco_version}.json"
    # )
    # coco_image_train_dir_path = (
    #     raw_coco_data_dir_path / "images" / f"train{coco_version}"
    # )
    # coco_image_val_dir_path = raw_coco_data_dir_path / "images" / f"val{coco_version}"

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

# text_embeddings_train_path = (
#     text_embeddings_dir_path / f"{text_embeddings_file_name}_train{coco_version}.pt"
# )
# text_embeddings_val_path = (
#     text_embeddings_dir_path / f"{text_embeddings_file_name}_val{coco_version}.pt"
# )

text_embeddings_nsd_path = (
    text_embeddings_dir_path / f"{text_embeddings_file_name}_nsd{coco_version}.pt"
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
    text_embeddings_nsd_var_path = (
        text_embeddings_dir_path
        / f"{text_embeddings_var_file_name}_nsd{coco_version}.pt"
    )


text_embeddings_meta_train_path = (
    text_embeddings_dir_path
    / f"{text_embeddings_meta_file_name}_train{coco_version}.json"
)
text_embeddings_meta_val_path = (
    text_embeddings_dir_path
    / f"{text_embeddings_meta_file_name}_val{coco_version}.json"
)

text_embeddings_nsd_meta_path = (
    text_embeddings_dir_path
    / f"{text_embeddings_meta_file_name}_nsd{coco_version}.json"
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

if eval_flag:
    evaluation_data_dir_path = (
        checkpoint_path.parent / "evaluation" / checkpoint_path.stem
    )
    evaluation_data_dir_path.mkdir(parents=True, exist_ok=True)
    evaluation_loss_path = evaluation_data_dir_path / "evaluation_loss.json"
    evaluation_alignment_data_path = evaluation_data_dir_path / "alignment_data.json"

    model_activation_dir_path = replace_subdir(
        checkpoint_path.parent,
        "checkpoints",
        "model_activations",
    )
    model_activation_dir_path = model_activation_dir_path / checkpoint_path.stem
    model_activation_dir_path.mkdir(parents=True, exist_ok=True)

    rsa_dir_path = replace_subdir(
        checkpoint_path.parent,
        "checkpoints",
        "rsa",
    )
    rsa_dir_path.mkdir(parents=True, exist_ok=True)

fmri_rdm_dir_path.mkdir(parents=True, exist_ok=True)
noise_ceiling_path = data_dir_path / "rsa" / "rsa_rdm_noise_ceiling_special515.json"


###### Functions ######
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(sentence_transformer_model_name)
    return _model
