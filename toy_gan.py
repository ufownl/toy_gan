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


if __name__ == "__main__":
    net_g = Generator(64)
    net_g.initialize(mx.init.Xavier())
    net_d = Discriminator(64)
    net_d.initialize(mx.init.Xavier())
    seeds = mx.nd.random_normal(shape=(4, 128, 1, 1))
    imgs = net_g(seeds)
    print(imgs)
    print(net_d(imgs))
