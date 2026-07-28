import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from experiments.analysis.roi_mask import get_roi_mask, ALL_ROIS
import nibabel as nb
from nilearn import datasets
from nilearn import plotting
import os
from matplotlib.colors import ListedColormap

ROI_BASE_COLOR = {
    "early": "yellowgreen",
    "midparietal": "salmon",
    "parietal": "indianred",
    "midlateral": "lightskyblue",
    "lateral": "royalblue",
    "midventral": "moccasin",
    "ventral": "gold",
}


def plot_diff_raincloud(
    values_array,
    value_type,
    model_names,
    ylabel,
    title=None,
    cosine_group="all",
    font_size=9,
):

    # LLM-guided minus Standard, paired per-sample, added only if both names are present
    diffs = []
    labels = []

    def add_diff(pos_name, neg_name, label):
        if pos_name in model_names and neg_name in model_names:
            diffs.append(
                values_array[model_names.index(pos_name)]
                - values_array[model_names.index(neg_name)]
            )
            labels.append(label)

    if value_type == "cosine_similarity":
        # cosine_group: "vs_standard" -> LLM-guided vs Standard (AE, VAE μ),
        #               "latent"      -> latent-space comparisons (z − μ, VAE(z) − AE),
        #               "all"         -> both groups in one figure
        if cosine_group in ("all", "vs_standard"):
            add_diff(
                "LLM-guided VAE (mu)",
                "Standard VAE (mu)",
                "VAE (μ)",
            )
            add_diff("LLM-guided AE", "Standard AE", "AE (z)")
        if cosine_group in ("all", "latent"):
            add_diff(
                "LLM-guided VAE (latent)",
                "LLM-guided VAE (mu)",
                "LLM-guided VAE\n(z − μ)",
            )
            add_diff(
                "LLM-guided VAE (latent)",
                "LLM-guided AE",
                "LLM-guided\n(VAE (z) − AE)",
            )
        if cosine_group not in ("all", "vs_standard", "latent"):
            raise ValueError(
                f"cosine_group should be 'all', 'vs_standard' or 'latent', got '{cosine_group}'"
            )
    elif value_type == "reconstruction_loss":
        add_diff(
            "LLM-guided VAE",
            "Standard VAE",
            "VAE",
        )
        add_diff("LLM-guided AE", "Standard AE", "AE")

    else:
        raise ValueError(
            f"value_type should be 'cosine_similarity' or 'reconstruction_loss', got '{value_type}'"
        )

    diff_df = pd.DataFrame(
        {
            "diff": np.concatenate(diffs),
            "model": sum(([label] * len(d) for label, d in zip(labels, diffs)), []),
        }
    )

    fig, ax = plt.subplots(figsize=(2 * len(labels), 4))
    sns.violinplot(
        data=diff_df,
        x="model",
        y="diff",
        inner=None,
        color="tab:grey",
        alpha=0.5,
        cut=0,
        ax=ax,
        split=True,
        linewidth=0,
    )
    # re-scale and shift the violin
    shrink_factor = 0.5
    shift_amount = 0.2
    for i, collection in enumerate(ax.collections):
        for path in collection.get_paths():
            verts = path.vertices
            verts[:, 0] = i - (verts[:, 0] - i) * shrink_factor
            verts[:, 0] += shift_amount

    sns.boxplot(
        data=diff_df,
        x="model",
        y="diff",
        color="white",
        width=0.1,
        whis=[0, 100],
        boxprops={"zorder": 2},
        linewidth=1.5,
        ax=ax,
    )

    # stripplot (rain) - shift it to the right so it clears the box/violin
    n_before_strip = len(ax.collections)
    sns.stripplot(
        data=diff_df,
        x="model",
        y="diff",
        color="tab:grey",
        alpha=0.5,
        jitter=0.2,
        size=0.5,
        ax=ax,
    )

    strip_shift = -0.25
    for collection in ax.collections[n_before_strip:]:
        offsets = collection.get_offsets()
        offsets[:, 0] += strip_shift
        collection.set_offsets(offsets)

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    plt.tight_layout()
    plt.show()


def plot_log_var_evolution(
    log_var_median_across_samples_array,
    log_var_rand_array,
    checkpoints,
    n_samples,
    n_dims,
    n_samples_to_plot,
    example_pos=20,
):
    epoch_labels = [f"Epoch {c.split('epoch')[1]}" for c in checkpoints]
    epoch_nums = [int(c.split("epoch")[1]) for c in checkpoints]

    epoch_colors = plt.cm.Blues(
        np.linspace(0.3, 1, len(epoch_labels))
    )  # color deepens as epoch increases

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 3)

    axes_row1 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_row2 = fig.add_subplot(gs[1, :])

    # sort latent dim based on log_var
    sort_idx = np.argsort(log_var_median_across_samples_array[-1])
    candidates = np.where(
        log_var_median_across_samples_array[1] + 2
        < log_var_median_across_samples_array[-1]
    )[0]
    mid_dim_id = candidates[0]
    mid_dim_sorted_id = int(np.where(sort_idx == mid_dim_id)[0][0])

    dim_ids = [sort_idx[example_pos], mid_dim_id, sort_idx[-example_pos]]
    dim_sorted_ids = [example_pos, mid_dim_sorted_id, n_dims - example_pos]
    dim_colors = ["tab:orange", "tab:orange", "tab:orange"]

    fig.suptitle(f"\n{n_samples_to_plot} random samples out of {n_samples}")
    for ax, dim_id, color, sorted_id in zip(
        axes_row1, dim_ids, dim_colors, dim_sorted_ids
    ):
        for sample in range(n_samples_to_plot):
            trajectory = log_var_rand_array[:, sample, dim_id]
            ax.plot(
                epoch_nums,
                trajectory,
                marker="o",
                alpha=0.3,
                color=color,
                linewidth=0.4,
                markersize=3,
            )
        ax.set_title(f"Latent dim {dim_id}\n(sorted position {sorted_id})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("log(variance)")
        ax.grid(True, alpha=0.3)

    # per-dim mean log_var
    for i, label in enumerate(epoch_labels):
        ax_row2.plot(
            log_var_median_across_samples_array[i][sort_idx],
            alpha=0.8,
            label=label,
            color=epoch_colors[i],
        )

    # highlight the dims shown above, using matching colors
    for x_pos, color in zip(dim_sorted_ids, dim_colors):
        ax_row2.axvline(x_pos, color=color, linestyle="--", alpha=0.7, zorder=0)

    ax_row2.set_xlabel("Latent dim (sorted by last epoch)")
    ax_row2.set_xlim(0, log_var_median_across_samples_array.shape[1])
    ax_row2.set_ylabel("Median log(variance)")
    ax_row2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax_row2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_rois(
    roi_defs_dir_path, hemisphere, map_type="infl", which_rois="streams", title=None
):
    """Visualise rois
    Parameters
    ----------
    roi_defs_dir_path : str
        Path to the directory with roi data
    hemisphere : str
        The hemisphere of the brain ('left' or 'right').
    map_type : str
        Brain map type
    which_rois: str
        only "rois" for now
    title: str
        figure title

    Returns
    -------
    view :
        An interactive brain surface map view object.

    """

    stream_rois = [r for r in ALL_ROIS if r != "all"]

    cmap = ListedColormap([ROI_BASE_COLOR[r] for r in stream_rois])

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    # check the size of left hem data
    try:
        lh_file = os.path.join(roi_defs_dir_path, f"lh.{which_rois}.mgz")
        mapping_lh = nb.load(lh_file).get_fdata().squeeze()
    except ValueError:
        lh_file = os.path.join(roi_defs_dir_path, f"lh.{which_rois}.npy")
        mapping_lh = np.load(lh_file, allow_pickle=True)

    n_left = len(mapping_lh)

    combined = None
    id2name = {}
    for i, roi in enumerate(stream_rois, start=1):
        mask = np.asarray(get_roi_mask(roi, roi_defs_dir_path))
        if combined is None:
            combined = np.zeros(len(mask))
        combined[mask] = i
        id2name[i] = roi

    # Split the left and right hemisphere
    sl = slice(0, n_left) if hemisphere == "left" else slice(n_left, None)
    surf_map = combined[sl]

    view = plotting.view_surf(
        surf_mesh=fsaverage[map_type + "_" + hemisphere],
        surf_map=surf_map,
        bg_map=fsaverage["sulc_" + hemisphere],
        threshold=0.5,
        cmap=cmap,
        symmetric_cmap=False,
        vmin=0.5,
        vmax=len(stream_rois) + 0.5,
        colorbar=False,
        title=title,
    )
    return view
