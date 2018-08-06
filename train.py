import os
import time
import argparse
import mxnet as mx
from dataset import load_dataset
from toy_gan import Generator, Discriminator

def train(max_epochs, learning_rate, threshold, batch_size, seed_size, filters, context):
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set = load_dataset(batch_size)

    net_g = Generator(filters)
    net_d = Discriminator(filters)
    loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    metric = mx.metric.CustomMetric(lambda label, pred: ((pred > 0.5) == label).mean())

    if os.path.isfile("model/toy_gan.generator.params"):
        net_g.load_parameters("model/toy_gan.generator.params", ctx=context)
    else:
        net_g.initialize(mx.init.Xavier(), ctx=context)

    if os.path.isfile("model/toy_gan.discriminator.params"):
        net_d.load_parameters("model/toy_gan.discriminator.params", ctx=context)
    else:
        net_d.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate, flush=True)
    trainer_g = mx.gluon.Trainer(net_g.collect_params(), "Adam", {
        "learning_rate": learning_rate,
        "clip_gradient": 5.0
    })
    trainer_d = mx.gluon.Trainer(net_d.collect_params(), "Adam", {
        "learning_rate": learning_rate,
        "clip_gradient": 5.0
    })

    if os.path.isfile("model/toy_gan.generator.state"):
        trainer_g.load_states("model/toy_gan.generator.state")

    if os.path.isfile("model/toy_gan.discriminator.state"):
        trainer_d.load_states("model/toy_gan.discriminator.state")

    real_label = mx.nd.ones((batch_size,), ctx=context)
    fake_label = mx.nd.zeros((batch_size,), ctx=context)

    print("Training...", flush=True)
    accuracy = 0.0
    for epoch in range(max_epochs):
        ts = time.time()

        training_L = 0.0
        training_batch = 0
        training_set.reset()
        metric.reset()

        for batch in training_set:
            training_batch += 1

            real = batch.data[0].as_in_context(context)
            seeds = mx.nd.random_normal(shape=(batch_size, seed_size, 1, 1), ctx=context)

            with mx.autograd.record():
                fake = net_g(seeds)

            if accuracy < threshold:
                with mx.autograd.record():
                    real_y = net_d(real)
                    real_L = loss(real_y, real_label)
                    metric.update([real_label], [real_y])
                    fake_y = net_d(fake.detach())
                    fake_L = loss(fake_y, fake_label)
                    metric.update([fake_label], [fake_y])
                    L = real_L + fake_L
                    L.backward()
                trainer_d.step(batch_size)
            else:
                real_y = net_d(real)
                real_L = loss(real_y, real_label)
                metric.update([real_label], [real_y])
                fake_y = net_d(fake.detach())
                fake_L = loss(fake_y, fake_label)
                metric.update([fake_label], [fake_y])
                L = real_L + fake_L
            _, accuracy = metric.get()
            dis_L = mx.nd.mean(L).asscalar()
            if dis_L != dis_L:
                raise ValueError()

            with mx.autograd.record():
                y = net_d(fake)
                L = loss(y, real_label)
                L.backward()
            trainer_g.step(batch_size)
            gen_L = mx.nd.mean(L).asscalar()
            if gen_L != gen_L:
                raise ValueError()
                
            training_L += dis_L + gen_L
            print("[Epoch %d  Batch %d]  dis_loss %.10f  gen_loss %.10f  average_loss %.10f  accuracy %.10f  elapsed %.2fs" % (
                epoch, training_batch, dis_L, gen_L, training_L / training_batch, accuracy, time.time() - ts
            ), flush=True)

        avg_L = training_L / training_batch
        print("[Epoch %d]  training_loss %.10f  accuracy %.10f  duration %.2fs" % (
            epoch + 1, avg_L, accuracy, time.time() - ts
        ), flush=True)

        net_g.save_parameters("model/toy_gan.generator.params")
        net_d.save_parameters("model/toy_gan.discriminator.params")
        trainer_g.save_states("model/toy_gan.generator.state")
        trainer_d.save_states("model/toy_gan.discriminator.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a toy_gan trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 0.0002)", type=float, default=0.0002)
    parser.add_argument("--threshold", help="set the accuracy threshold to pause the training of discriminator (default: 0.6)", type=float, default=0.6)
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
                threshold = args.threshold,
                batch_size = 256,
                seed_size = 128,
                filters = 64,
                context = context
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
