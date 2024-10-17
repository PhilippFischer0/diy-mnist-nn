import numpy as np
import matplotlib.pyplot as plt

# from https://yann.lecun.com/exdb/mnist/
def parse_mnist_data(
        idx_file_training_samples: str,
        idx_file_training_labels: str,
        idx_file_test_samples: str,
        idx_file_test_labels: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    
    training_samples = parse_mnist_images(idx_file_training_samples)
    training_labels = parse_mnist_labels(idx_file_training_labels)

    test_samples = parse_mnist_images(idx_file_test_samples)
    test_labels = parse_mnist_labels(idx_file_test_labels)

    return tuple[training_samples, training_labels, test_samples, test_labels]

def parse_mnist_images(idx_file_path: str) -> np.ndarray:
    f = open(idx_file_path, 'rb')

    magic_num = int.from_bytes(f.read(2), 'big')
    ftype = int.from_bytes(f.read(1), 'big')
    num_dim = int.from_bytes(f.read(1), 'big')
    num_img = int.from_bytes(f.read(4), 'big')
    num_rows = int.from_bytes(f.read(4), 'big')
    num_cols = int.from_bytes(f.read(4), 'big')

    data = f.read()
    out = np.ndarray((num_img, num_rows, num_cols), np.uint8, data)
    print(out.shape)
    return out

def parse_mnist_labels(idx_file_path: str) -> np.ndarray:
    f = open(idx_file_path, 'rb')
    
    magic_num = int.from_bytes(f.read(2), 'big')
    ftype = int.from_bytes(f.read(1), 'big')
    num_dim = int.from_bytes(f.read(1), 'big')
    num_item = int.from_bytes(f.read(4), 'big')

    data = f.read()
    out = np.ndarray((num_item, 1), np.uint8, data)
    print(out.shape)
    return out

def plot_image(img: np.ndarray) -> plt.figure:
    assert len(img.shape) == 2, "input must be 2-dimensional (single image)"

    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()

class FeedForward:
    def __init__(self, fan_in: int, num_hidden: int, fan_out: int) -> None:
        # define matrices
        self.layer_1_matrix = np.random.uniform(-1, 1, (fan_in, num_hidden)).astype(np.float32)
        self.layer_2_matrix = np.random.uniform(-1, 1, (num_hidden, fan_out)).astype(np.float32)
        # define bias
        self.bias_1 = np.random.uniform(-1, 1, num_hidden).astype(np.float32)
        self.bias_2 = np.random.uniform(-1, 1, fan_out).astype(np.float32)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)/np.sum(np.exp(x))

    def __call__(self, x: np.ndarray) -> np.ndarray:

        # multiplication witht matrix 1 (weights) -> add bias
        xl1 = x @ self.layer_1_matrix + self.bias_1
        # activation function (np.tanh) anwenden
        xl1a = np.tanh(xl1)
        # multiplikation mit matrix 2 (weights) -> bias addieren
        xl2 = xl1a @ self.layer_2_matrix + self.bias_2
        # normalisierung mit softmax
        xl2n = self.softmax(xl2)

        return xl2n

def foo():
    print("hallo blub")