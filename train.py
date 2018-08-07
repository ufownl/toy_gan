import os
import time
import argparse
import mxnet as mx
from dataset import load_dataset
from toy_gan import Generator, Discriminator, WassersteinLoss

def train(max_epochs, learning_rate, batch_size, seed_size, filters, context):
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set = load_dataset(batch_size)

    net_g = Generator(filters)
    net_d = Discriminator(filters)
    loss = WassersteinLoss()

    if os.path.isfile("model/toy_gan.generator.params"):
        net_g.load_parameters("model/toy_gan.generator.params", ctx=context)
    else:
        net_g.initialize(mx.init.Xavier(), ctx=context)

    if os.path.isfile("model/toy_gan.discriminator.params"):
        net_d.load_parameters("model/toy_gan.discriminator.params", ctx=context)
    else:
        net_d.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate, flush=True)
    trainer_g = mx.gluon.Trainer(net_g.collect_params(), "RMSProp", {
        "learning_rate": learning_rate,
        "clip_gradient": 5.0
    })
    trainer_d = mx.gluon.Trainer(net_d.collect_params(), "RMSProp", {
        "learning_rate": learning_rate,
        "clip_weights": 0.01,
        "clip_gradient": 5.0
    })

    if os.path.isfile("model/toy_gan.generator.state"):
        trainer_g.load_states("model/toy_gan.generator.state")

    if os.path.isfile("model/toy_gan.discriminator.state"):
        trainer_d.load_states("model/toy_gan.discriminator.state")

    print("Training...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        training_L = 0.0
        training_batch = 0
        training_set.reset()

        for batch in training_set:
            training_batch += 1

            real = batch.data[0].as_in_context(context)
            seeds = mx.nd.random_normal(shape=(batch_size, seed_size, 1, 1), ctx=context)

            with mx.autograd.record():
                real_y = net_d(real)
                fake = net_g(seeds)
                fake_y = net_d(fake.detach())
                L = loss(fake_y, real_y)
                L.backward()
            trainer_d.step(batch_size)
            batch_L = mx.nd.mean(L).asscalar()
            if batch_L != batch_L:
                raise ValueError()

            with mx.autograd.record():
                y = net_d(fake)
                L = loss(y)
                L.backward()
            trainer_g.step(batch_size)
            gen_L = mx.nd.mean(L).asscalar()
            if gen_L != gen_L:
                raise ValueError()
                
            training_L += batch_L
            print("[Epoch %d  Batch %d]  batch_loss %.10f  gen_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                epoch, training_batch, batch_L, gen_L, training_L / training_batch, time.time() - ts
            ), flush=True)

        avg_L = training_L / training_batch
        print("[Epoch %d]  training_loss %.10f  duration %.2fs" % (
            epoch + 1, avg_L, time.time() - ts
        ), flush=True)

        net_g.save_parameters("model/toy_gan.generator.params")
        net_d.save_parameters("model/toy_gan.discriminator.params")
        trainer_g.save_states("model/toy_gan.generator.state")
        trainer_d.save_states("model/toy_gan.discriminator.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a toy_gan trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.00005)", type=float, default=0.00005)
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    while True:
        try:
            train(
                max_epochs = args.max_epochs,
                learning_rate = args.learning_rate,
                batch_size = 256,
                seed_size = 128,
                filters = 64,
                context = context
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
