import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def binary_parse_mnist_data(
    idx_file_training_samples: str,
    idx_file_training_labels: str,
    idx_file_test_samples: str,
    idx_file_test_labels: str,
    number_1: int,
    number_2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    training_samples = parse_mnist_images(idx_file_training_samples)
    training_labels = parse_mnist_labels(idx_file_training_labels)

    test_samples = parse_mnist_images(idx_file_test_samples)
    test_labels = parse_mnist_labels(idx_file_test_labels)

    return training_samples, training_labels, test_samples, test_labels


def parse_mnist_images(idx_file_path: str) -> np.ndarray:
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
