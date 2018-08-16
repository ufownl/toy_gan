import time
import argparse
import mxnet as mx
import matplotlib.pyplot as plt
from dataset import visualize
from toy_gan import Generator

def test(batch_size, seed_size, filters, context):
    mx.random.seed(int(time.time()))

    net_g = Generator(filters)
    net_g.load_parameters("model/toy_gan.generator.params", ctx=context)

    seeds = mx.nd.random_normal(shape=(batch_size, seed_size, 1, 1), ctx=context)
    imgs = net_g(seeds)
    for i in range(imgs.shape[0]):
        plt.subplot(imgs.shape[0] // 8 + 1, 8, i + 1)
        visualize(imgs[i])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a toy_gan tester.")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    test(
        batch_size = 32,
        seed_size = 128,
        filters = 64,
        context = context
    )

