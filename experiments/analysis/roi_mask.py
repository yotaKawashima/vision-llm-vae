import os
import numpy as np
import nibabel as nb

ALL_ROIS = [
    "early",
    "midventral",
    "ventral",
    "midlateral",
    "lateral",
    "midparietal",
    "parietal",
]


def get_roi_mapping(
    roi_defs_dir_path: str,
    which_rois: str = "streams",
) -> (np.ndarray, dict):
    """Get the mapping between voxels and roi labels from the specified directory.
    Parameters
    ----------
    roi_defs_dir: str
        Directory containing the ROI definitions.
    which_rois: str
        Name of the NSD ROI to load. Default "streams".
    Returns
    -------
    maskdata: np.ndarray
        The loaded ROI data.
    roi_id2name: dict
        Mapping of ROI IDs to names.
    """
    roi_names_file = os.path.join(roi_defs_dir_path, f"{which_rois}.mgz.ctab")

    try:
        with open(roi_names_file) as f:
            # get ROI names automatically. If you don't have the .ctab file
            # you can also enter them by hand. 0 is always "Unknown")
            roi_id2name = {int(x[0]): x[2:-1] for x in f}

    except ValueError:
        print(
            f"roi_names_file not found. Requested {roi_names_file}. Using {which_rois} as single ROI name."
        )
        roi_id2name = {0: "Unknown"}
        roi_id2name[1] = which_rois

    # load the roi masks
    try:
        lh_file = os.path.join(roi_defs_dir_path, f"lh.{which_rois}.mgz")
        rh_file = os.path.join(roi_defs_dir_path, f"rh.{which_rois}.mgz")
        mapping_lh = nb.load(lh_file).get_fdata().squeeze()
        mapping_rh = nb.load(rh_file).get_fdata().squeeze()
    except ValueError:
        lh_file = os.path.join(roi_defs_dir_path, f"lh.{which_rois}.npy")
        rh_file = os.path.join(roi_defs_dir_path, f"rh.{which_rois}.npy")
        mapping_lh = np.load(lh_file, allow_pickle=True)
        mapping_rh = np.load(rh_file, allow_pickle=True)

    mapping = np.hstack((mapping_lh, mapping_rh))

    return mapping, roi_id2name


def get_roi_mask(
    roi: str,
    roi_defs_dir_path: str,
    which_rois: str = "streams",
) -> np.ndarray:
    """Get a mask for a target roi.)
    Parameters
    ----------
    roi: str
        Name of the target roi.
    which_rois: str
        Name of the NSD ROI to load.
    roi_defs_dir: str
        Directory containing the ROI definitions.
    Returns
    -------
    mask: np.ndarray of boolean
        The mask for the specified ROI.
    """
    if which_rois != "streams":
        print("Currently, only streams ROIs are supported.")
        ValueError(f"Invalid which_rois: {which_rois}. Valid options are: 'streams'.")

    if roi in ALL_ROIS:
        print(f"Loading ROI mask for {roi}...")
    else:
        print("Currently, only streams ROIs are supported.")
        ValueError(f"Invalid ROI name: {roi}. Valid options are: {ALL_ROIS}.")

    # load roi mapping
    mapping, roi_id2name = get_roi_mapping(roi_defs_dir_path, which_rois)

    # check voxel for the target roi
    target_roi_id = [key for key, value in roi_id2name.items() if value == roi][0]
    mask = mapping == target_roi_id

    return mask
