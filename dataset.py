import os
import tarfile
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

def load_image(path):
    with open(path, "rb") as f:
        buf = f.read()
    return mx.image.imdecode(buf)

def cook_image(img, size):
    img = mx.image.resize_short(img, min(size))
    img, _ = mx.image.center_crop(img, size)
    return img.astype("float32") / 127.5 - 1.0

def load_dataset(batch_size, image_size=(64, 64)):
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    data_path = "data"
    if not os.path.exists(data_path):
        data_file = mx.gluon.utils.download(lfw_url)
        with tarfile.open(data_file) as tar:
            os.makedirs(data_path)
            tar.extractall(path=data_path)
    img_path = [os.path.join(path, f) for path, _, files in os.walk(data_path) for f in files]
    imgs = [cook_image(load_image(img), image_size).T.expand_dims(0) for img in img_path]
    return mx.io.NDArrayIter(mx.nd.concat(*imgs, dim=0), batch_size=batch_size, shuffle=True)

def visualize(img):
   plt.imshow(((img.T + 1.0) * 127.5).asnumpy().astype(np.uint8))
   plt.axis("off")


if __name__ == "__main__":
    dataset = load_dataset(32)
    batch = next(dataset).data[0]
    print("batch preview: ", batch)
    for i in range(batch.shape[0]):
        plt.subplot(batch.shape[0] // 8 + 1, 8, i + 1)
        visualize(batch[i])
    plt.show()
