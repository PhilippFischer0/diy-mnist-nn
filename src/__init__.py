import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_number_of_samples(
    images: np.ndarray, labels: np.ndarray, number_of_images_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    :params np.ndarray images: array of multiple images
    :params np.ndarray labels: array of the labels for the images
    :params int number_of_images_per_class: number of images that should be rreturned for each class
    :returns: Tuple of arrays with 2*number_of_images_per_class images and labels that are equally distributed
    """
    sampled_images = []
    sampled_labels = []

    labels = labels.flatten()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        img_label = images[labels == label]
        lab_label = labels[labels == label]

        if len(img_label) >= number_of_images_per_class:
            idx = np.random.choice(
                len(img_label), number_of_images_per_class, replace=False
            )
            sampled_images.append(img_label[idx])
            sampled_labels.append([[int(lab)] for lab in lab_label[idx]])
        else:
            raise ValueError(
                f"Not enough Images of label {label} to equally distribute {number_of_images_per_class} images"
            )

    sampled_images = np.concatenate(sampled_images, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0)
    return sampled_images, sampled_labels


def binary_parse_mnist_data(
    idx_file_training_samples: str,
    idx_file_training_labels: str,
    idx_file_test_samples: str,
    idx_file_test_labels: str,
    number_1: int,
    number_2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :params str idx_file_training_samples: file path to the training samples
    :params str idx_file_training_labels: file path to the training labels
    :params str idx_file_test_samples: file path to the testing samples
    :params str idx_file_test_labels: file path to the testing labels
    :params int number_1: first number between 0-9 that should be loaded from the MNIST dataset
    :params int number_2: second number between 0-9 that should be loaded from the MNIST dataset
    :returns: tuple of arrays with only the given numbers present
    """
    # get the data
    training_samples, training_labels, test_samples, test_labels = parse_mnist_data(
        idx_file_training_samples,
        idx_file_training_labels,
        idx_file_test_samples,
        idx_file_test_labels,
    )
    # filter only two numbers with a mask
    training_mask = (training_labels.flatten() == number_1) | (
        training_labels.flatten() == number_2
    )
    filtered_training_labels = training_labels[training_mask]
    filtered_training_samples = training_samples[training_mask]

    test_mask = (test_labels.flatten() == number_1) | (
        test_labels.flatten() == number_2
    )
    filtered_test_labels = test_labels[test_mask]
    filtered_test_samples = test_samples[test_mask]

    # downscale the samples to lessen the computational effort
    downscaled_training_samples = np.array(
        [
            Image.fromarray(train_img).resize((10, 10), Image.Resampling.LANCZOS)
            for train_img in filtered_training_samples
        ]
    )
    downscaled_testing_samples = np.array(
        [
            Image.fromarray(test_img).resize((10, 10), Image.Resampling.LANCZOS)
            for test_img in filtered_test_samples
        ]
    )

    # Normalisierung
    downscaled_training_samples = downscaled_training_samples / 255
    downscaled_testing_samples = downscaled_testing_samples / 255

    return (
        downscaled_training_samples,
        filtered_training_labels,
        downscaled_testing_samples,
        filtered_test_labels,
    )


# from https://yann.lecun.com/exdb/mnist/
def parse_mnist_data(
    idx_file_training_samples: str,
    idx_file_training_labels: str,
    idx_file_test_samples: str,
    idx_file_test_labels: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :params str idx_file_training_samples: file path to the training samples
    :params str idx_file_training_labels: file path to the training labels
    :params str idx_file_test_samples: file path to the testing samples
    :params str idx_file_test_labels: file path to the testing labels
    :returns: tuple of arrays with the MNIST dataset
    """

    training_samples = parse_mnist_images(idx_file_training_samples)
    training_labels = parse_mnist_labels(idx_file_training_labels)

    test_samples = parse_mnist_images(idx_file_test_samples)
    test_labels = parse_mnist_labels(idx_file_test_labels)

    return training_samples, training_labels, test_samples, test_labels


def parse_mnist_images(idx_file_path: str) -> np.ndarray:
    """
    :params str idx_file_path: path to the image file
    :returns: array of images
    """
    with open(idx_file_path, "rb") as f:

        # read magic number
        f.read(4)
        num_img = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        data = f.read()
        out = np.ndarray((num_img, num_rows, num_cols), np.uint8, data)
        return out


def parse_mnist_labels(idx_file_path: str) -> np.ndarray:
    """
    :params str idx_file_path: path to the label file
    :returns: array of labels
    """
    with open(idx_file_path, "rb") as f:

        # read magic number
        f.read(4)
        num_item = int.from_bytes(f.read(4), "big")

        data = f.read()
        out = np.ndarray((num_item, 1), np.uint8, data)
        return out


def plot_image(img: np.ndarray) -> plt.Figure:
    assert len(img.shape) == 2, "input must be 2-dimensional (single image)"

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(img, cmap="gray")

    plt.close()
    return fig
