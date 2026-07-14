# LLM-informed Variational Autoencoder

## Overview

This repository explores whether guiding a Variational Autoencoder (VAE) with a Large
Language Model (LLM) improves its representational alignment with the human visual system.
We introduced an **LLM-informed VAE** whose posterior means are constrained by LLM
embeddings of image captions, and evaluated it against functional magnetic resonance imaging (fMRI) data from natural scene viewing.

![VAE overview](vae_overview.png)

The code lets you train the model family (`encoder`, `ae`, `beta_vae`) under a standard or
LLM-alignment objective, and evaluate how well the learned latents align with higher
visual areas. See below for how to configure training and evaluation via `config.py`.

> **Note:** Everything below this point is still a work in progress.

## Environment (Apptainer)

The environment is packaged as an [Apptainer](https://apptainer.org/) container, defined
in `image_mini.def`.

Build the container image (`image_mini.sif`) with:

```bash
apptainer build --fakeroot --force image_mini.sif image_mini.def
```

## Configuration Parameters (`config.py`)

### Training
You set `eval_flag = False`.
`encoder` is trained from scratch. `ae` reuses the `encoder`'s weights, freezes the
encoder, and trains only the decoder. `beta_vae` initializes both its encoder and decoder
from the `ae`'s weights, then trains the whole model. Set the corresponding checkpoint
path accordingly:
| Model             | `encoder_checkpoint` | `ae_checkpoint` | `vae_checkpoint` |
|:-----------------:|:--------------------:|:---------------:|:----------------:|
| `encoder`         | `False`              | `False`         | `False`          |
| `ae`              | `True`               | `False`         | `False`          |
| `beta_vae`        | `False`              | `True`          | `False`           

### Evaluation

You set `eval_flag = True`.
Unlike training, the checkpoint you provide is the one for the model you are evaluating,
so the variables are set as follows. Note that for the initialized model (`beta_vae`
(init.)), you instead provide the `ae`'s weights.
| Model             | `encoder_checkpoint` | `ae_checkpoint` | `vae_checkpoint` |
|:-----------------:|:--------------------:|:---------------:|:----------------:|
| `encoder`         | `False`              | `False`         | `False`          |
| `ae`              | `False`              | `True`          | `False`          |
| `beta_vae`        | `False`              | `False`         | `True`           |
| `beta_vae` (init.)| `False`              | `True`          | `False`          |



