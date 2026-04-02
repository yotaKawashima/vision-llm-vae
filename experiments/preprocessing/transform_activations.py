from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA


def pca(train_activations, test_activations, n_components=768):
    """
    Apply PCA to the activations.

    Parameters
    ----------
    train_activations: np.ndarray
        A 2D array of shape (num_samples, num_features) containing the training activations to be transformed.
    test_activations: np.ndarray
        A 2D array of shape (num_samples, num_features) containing the test activations to be transformed.
    n_components : int, optional
        The number of principal components to keep. Default is 768.

    Returns
    -------
    np.ndarray, np.ndarray
        The PCA-transformed training and test activations. Each is a 2D array
        of shape (num_samples, n_components), where num_samples is the number of
        samples in the respective dataset.
    """
    # Standardize the activations before applying PCA
    scaler = StandardScaler()
    train_activations_scaled = scaler.fit_transform(train_activations)
    test_activations_scaled = scaler.transform(test_activations)

    # Apply PCA
    pca = IncrementalPCA(n_components=n_components, batch_size=1000)
    train_activations_pca = pca.fit_transform(train_activations_scaled)
    test_activations_pca = pca.transform(test_activations_scaled)

    return train_activations_pca, test_activations_pca
