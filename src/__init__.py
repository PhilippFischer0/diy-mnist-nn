import numpy as np
import matplotlib.pyplot as plt


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


def softmax(x: np.ndarray) -> np.ndarray:
    # numerische Stabilität um overflow zu vermeiden
    exp_element = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_element / np.sum(exp_element, axis=1, keepdims=True)


def softmax_deriv(x: np.ndarray) -> np.ndarray:
    exp_element = np.exp(x - np.max(x, axis=1, keepdims=True))
    return (
        exp_element
        / np.sum(exp_element, axis=1)
        * (1 - exp_element / np.sum(exp_element, axis=1))
    )


def tanh_deriv(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.pow(np.tanh(x), 2)


def get_loss(y_pred: np.ndarray, y: np.ndarray) -> float:
    # cross entropy
    loss = 0

    for i in range(len(y_pred)):
        # calculate loss for each predicted value and add up
        loss += -1 * y[i] * np.log(y_pred[i])
    return loss


class FeedForward:

    def __init__(self, fan_in: int, num_hidden: int, fan_out: int) -> None:
        # define matrices
        self.layer_1_matrix = np.random.uniform(-1, 1, (fan_in, num_hidden)).astype(
            np.float32
        )
        self.layer_2_matrix = np.random.uniform(-1, 1, (num_hidden, fan_out)).astype(
            np.float32
        )
        # define bias
        self.bias_1 = np.random.uniform(-1, 1, num_hidden).astype(np.float32)
        self.bias_2 = np.random.uniform(-1, 1, fan_out).astype(np.float32)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensure the input is 2-dimensional
        if x.ndim == 1:
            x = x.reshape(1, -1)

        assert (
            x.shape[1] == self.layer_1_matrix.shape[0]
        ), "Input dimensions dont match with first layer"

        # multiplication witht matrix 1 (weights) -> add bias
        x = x @ self.layer_1_matrix + self.bias_1
        # activation function (np.tanh) anwenden
        x = np.tanh(x)
        # multiplikation mit matrix 2 (weights) -> bias addieren
        x = self.x @ self.layer_2_matrix + self.bias_2
        # normalisierung mit softmax
        x = softmax(x)

        return x
