import numpy as np


def compute_rdm_correlation(features):
    """
    Computes an RDM from features based on correlation.

    Parameters
    ----------
    features : np.ndarray
        A 2D array of shape (n_samples, n_features) containing the features for which to compute the RDM.
    Returns
        np.ndarray
        A 2D array of shape (n_samples, n_samples) containing the RDM values, where RDM[i, j] is the correlation distance between features[i] and features[j].
    -------

    """
    # subtract mean from each feature vector
    f = features - features.mean(axis=1, keepdims=True)
    # normalize feature vectors to unit length
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    corr_matrix = np.dot(f, f.T)
    # corr_matrix can have values slightly outside the range [-1, 1] due to numerical precision issues
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return 1 - corr_matrix
