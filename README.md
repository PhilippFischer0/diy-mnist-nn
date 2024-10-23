# diy-mnist-nn

# Dev Setup

Download the files via:

```bash
mkdir data
cd data
curl -Lso "./data/train_img.gz" https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -Lso "./data/train_label.gz" https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -Lso "./data/test_img.gz" https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -Lso "./data/test_label.gz" https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *
```

in the directory of the files and rename to .idx

Arm git filter to remove notebook output:

```bash
git config --local include.path ../.gitconfig
```
