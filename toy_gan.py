import mxnet as mx

class Generator(mx.gluon.nn.Block):
    def __init__(self, filters, **kwargs):
        super(Generator, self).__init__(**kwargs)
        with self.name_scope():
            self._net = mx.gluon.nn.Sequential()
            # input size: (batch_size, Z, 1, 1)
            self._net.add(mx.gluon.nn.Conv2DTranspose(filters * 8, 4, 1, 0,  use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters * 8, 4, 4)
            self._net.add(mx.gluon.nn.Conv2DTranspose(filters * 4, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters * 4, 8, 8)
            self._net.add(mx.gluon.nn.Conv2DTranspose(filters * 2, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters * 2, 16, 16)
            self._net.add(mx.gluon.nn.Conv2DTranspose(filters, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters, 32, 32)
            self._net.add(mx.gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.Activation("tanh"))
            # output size: (batch_size, 3, 64, 64)

    def forward(self, seeds):
        return self._net(seeds)


class Discriminator(mx.gluon.nn.Block):
    def __init__(self, filters, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self._net = mx.gluon.nn.Sequential()
            # input size: (batch_size, 3, 64, 64)
            self._net.add(mx.gluon.nn.Conv2D(filters, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters, 32, 32)
            self._net.add(mx.gluon.nn.Conv2D(filters * 2, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 2, 16, 16)
            self._net.add(mx.gluon.nn.Conv2D(filters * 4, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 4, 8, 8)
            self._net.add(mx.gluon.nn.Conv2D(filters * 8, 4, 2, 1, use_bias=False))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 8, 4, 4)
            self._net.add(mx.gluon.nn.Conv2D(1, 4, 1, 0, use_bias=False))
            # output size: (batch_size, 1, 1, 1)

    def forward(self, images):
        y = self._net(images)
        return y.reshape((-1,))


class GeneratorLoss(mx.gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(GeneratorLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, fake_y):
        return F.mean(fake_y, axis=self._batch_axis, exclude=True)


class DiscriminatorLoss(mx.gluon.loss.Loss):
    def __init__(self, lmda, batch_axis=0, **kwargs):
        super(DiscriminatorLoss, self).__init__(None, batch_axis, **kwargs)
        self._lambda = lmda

    def hybrid_forward(self, F, real_y, fake_y, real_x, fake_x):
        delta = F.sum(F.abs(real_x - fake_x), axis=self._batch_axis, exclude=True)
        loss = F.relu(real_y - fake_y + delta * self._lambda)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


if __name__ == "__main__":
    net_g = Generator(64)
    net_g.initialize(mx.init.Xavier())
    net_d = Discriminator(64)
    net_d.initialize(mx.init.Xavier())
    loss_g = GeneratorLoss()
    loss_d = DiscriminatorLoss(0.0002)
    real = mx.nd.zeros((4, 3, 64, 64))
    real_y = net_d(real)
    print("real_y: ", real_y)
    seeds = mx.nd.random_normal(shape=(4, 128, 1, 1))
    fake = net_g(seeds)
    print("fake: ", fake)
    fake_y = net_d(fake)
    print("fake_y: ", fake_y)
    print("loss_g: ", loss_g(fake_y))
    print("loss_d: ", loss_d(real_y, fake_y, real, fake))
