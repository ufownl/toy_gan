import os
import time
import argparse
import mxnet as mx
from dataset import load_dataset
from toy_gan import Generator, Discriminator

def train(batch_size, seed_size, filters, context):
    mx.random.seed(int(time.time()))

    print("Loading dataset...", flush=True)
    training_set = load_dataset(batch_size)

    net_g = Generator(filters)
    net_d = Discriminator(filters)
    loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
    metric = mx.metric.CustomMetric(lambda label, pred: ((pred > 0.5) == label).mean())

    if os.path.isfile("model/toy_gan.ckpt"):
        with open("model/toy_gan.ckpt", "r") as f:
            ckpt_lines = f.readlines()
        ckpt_argv = ckpt_lines[-1].split()
        epoch = int(ckpt_argv[0])
        best_L = float(ckpt_argv[1])
        learning_rate = float(ckpt_argv[2])
        epochs_no_progress = int(ckpt_argv[3])
        net_g.load_parameters("model/toy_gan.generator.params", ctx=context)
        net_d.load_parameters("model/toy_gan.discriminator.params", ctx=context)
    else:
        epoch = 0
        best_L = float("Inf")
        epochs_no_progress = 0
        learning_rate = 0.0002
        net_g.initialize(mx.init.Xavier(), ctx=context)
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

    real_label = mx.nd.ones((batch_size,), ctx=context)
    fake_label = mx.nd.zeros((batch_size,), ctx=context)

    print("Training...", flush=True)
    while learning_rate >= 1e-8:
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
                real_y = net_d(real)
                real_L = loss(real_y, real_label)
                metric.update([real_label], [real_y])
                fake = net_g(seeds)
                fake_y = net_d(fake.detach())
                fake_L = loss(fake_y, fake_label)
                metric.update([fake_label], [fake_y])
                L = real_L + fake_L
                L.backward()
            trainer_d.step(batch_size)
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
            print("[Epoch %d  Batch %d]  dis_loss %.10f  gen_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                epoch, training_batch, dis_L, gen_L, training_L / training_batch, time.time() - ts
            ), flush=True)

        epoch += 1
        avg_L = training_L / training_batch
        _, accuracy = metric.get()

        print("[Epoch %d]  learning_rate %.10f  training_loss %.10f  accuracy %.10f  epochs_no_progress %d  duration %.2fs" % (
            epoch, learning_rate, avg_L, accuracy, epochs_no_progress, time.time() - ts
        ), flush=True)

        if avg_L < best_L:
            best_L = avg_L
            epochs_no_progress = 0
            net_g.save_parameters("model/toy_gan.generator.params")
            net_d.save_parameters("model/toy_gan.discriminator.params")
            with open("model/toy_gan.ckpt", "a") as f:
                f.write("%d %.10f %.10f %d\n" % (epoch, best_L, learning_rate, epochs_no_progress))
        elif epochs_no_progress < 10:
            epochs_no_progress += 1
        else:
            epochs_no_progress = 0
            learning_rate *= 0.5
            trainer_g.set_learning_rate(learning_rate)
            trainer_d.set_learning_rate(learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a toy_gan trainer.")
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
                batch_size = 256,
                seed_size = 128,
                filters = 64,
                context = context
            )
            break;
        except ValueError:
            print("Oops! The value of loss become NaN...")
