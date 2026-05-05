import numpy as np
from scipy.stats import spearmanr


def correlation_between_rdm(
    rdm1: np.ndarray, rdm2: np.ndarray, apply_upper_triangle: bool = True
) -> float:
    """
    Compute the Spearman correlation between two RDMs.

    Parameters
    ----------
    rdm1: np.ndarray
        The first RDM.
    rdm2: np.ndarray
        The second RDM.
    apply_upper_triangle: bool
        Whether to extract the upper triangle of the RDMs before computing the correlation. Defaults to True.

    Returns
    -------
    float
        The Spearman correlation between the two RDMs.
    """
    if apply_upper_triangle:
        # extract upper triangle of the RDMs
        rdm1_vectorized = rdm1[np.triu_indices_from(rdm1, k=1)]
        rdm2_vectorized = rdm2[np.triu_indices_from(rdm2, k=1)]
    else:
        rdm1_vectorized = rdm1 if rdm1.ndim == 1 else rdm1.flatten()
        rdm2_vectorized = rdm2 if rdm2.ndim == 1 else rdm2.flatten()

    # compute the Spearman correlation
    corr, _ = spearmanr(rdm1_vectorized, rdm2_vectorized)

    return corr


def compute_rdm_correlation(features):
    """
    Computes an RDM from features based on correlation.

    Parameters
    ----------
    features : np.ndarray
        A 2D array of shape (n_samples, n_features) containing the features for which to compute the RDM.

    Returns
    -------
        np.ndarray
        A 2D array of shape (n_samples, n_samples) containing the RDM values, where RDM[i, j] is the correlation distance between features[i] and features[j].
    """
    # subtract mean from each feature vector
    f = features - features.mean(axis=1, keepdims=True)
    # normalize feature vectors to unit length
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    corr_matrix = np.dot(f, f.T)
    # corr_matrix can have values slightly outside the range [-1, 1] due to numerical precision issues
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return 1 - corr_matrix
