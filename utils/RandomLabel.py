import numpy as np


def shuffle_labels(labels, seed=None):
    """
    Randomly shuffle label values to break label-data correspondence.

    Args:
        labels (np.ndarray): Label array to shuffle.
        seed (int | None): Random seed for reproducibility.

    Returns:
        np.ndarray: Shuffled label array with the same shape as input.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return shuffled
