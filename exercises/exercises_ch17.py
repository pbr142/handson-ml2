from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras


def get_cifar10_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Get and Format CIFAR10 data

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]: Returns (X_train, y_train), (X_test, y_test)
        Formated as float32
    """    
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = (X_train / 255.).astype('float32')
    X_test = (X_test / 255.).astype('float32')
    return (X_train, y_train), (X_test, y_test)


def get_stratified_sample(n: int, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get a sample of training data with stratified sampling

    Args:
        n (int): Number of samples to return, has to be less than len(X)
        X (np.array): Features to sample
        y (np.array): Target values

    Returns:
        Tuple[np.array, np.array]: Samples from X and y of length n
    """

    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=42)
    idx = sss.split(X, y)
    idx_train, _ = idx.__next__()
    X_small = X[idx_train]
    y_small = y[idx_train]

    return X_small, y_small


def plot_image_and_reconstruction(X: np.ndarray, model: keras.models.Model, idx: Optional[int]=None, cmap: Optional[str]='binary') -> None:
    """Plot an image and its reconstruction from an autoencoder model

    Args:
        X (np.ndarray): Training data of autoencoder
        model (keras.models.Model): Autoencoder
        idx (Optional[int], optional): The index of the training sample to plot. Defaults to None, in which case a random index is chosen.  
        cmap (Optional[str], optional): The image mapping for pyplot. Defaults to 'binary'.
    """    
    idx = idx or np.random.choice(len(X))
    
    image = X[idx]
    [reconstruction] = model.predict(image[np.newaxis,:])
    difference = np.abs(image - reconstruction)
    
    fig = plt.figure(figsize=(3 * 1.5, 3))
    plt.subplot(1,3,1)
    plt.imshow(image, cmap=cmap)
    plt.subplot(1,3,2)
    plt.imshow(reconstruction, cmap=cmap)
    plt.subplot(1,3,3)
    plt.imshow(difference, cmap=cmap)
    
    plt.show()
