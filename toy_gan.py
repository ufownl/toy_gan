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


class SNConv2D(mx.gluon.nn.Block):
    def __init__(self, channels, kernel_size, strides, padding, in_channels, epsilon=1e-8, **kwargs):
        super(SNConv2D, self).__init__(**kwargs)

        self._channels = channels
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._epsilon = epsilon

        with self.name_scope():
            self._weight = self.params.get("weight", shape=(channels, in_channels, kernel_size, kernel_size))
            self._u = self.params.get("u", init=mx.init.Normal(), shape=(1, channels))

    def _spectral_norm(self, ctx):
        w = self.params.get("weight").data(ctx)
        w_mat = w.reshape((w.shape[0], -1))
        v = mx.nd.L2Normalization(mx.nd.dot(self._u.data(ctx), w_mat))
        u = mx.nd.L2Normalization(mx.nd.dot(v, w_mat.T))
        self.params.setattr("u", u)
        sigma = mx.nd.sum(mx.nd.dot(u, w_mat) * v)
        if sigma == 0:
            sigma = self._epsilon
        return w / sigma

    def forward(self, x):
        return mx.nd.Convolution(
            data = x,
            weight = self._spectral_norm(x.context),
            kernel = (self._kernel_size, self._kernel_size),
            stride = (self._strides, self._strides),
            pad = (self._padding, self._padding),
            num_filter = self._channels,
            no_bias = True
        )


class Discriminator(mx.gluon.nn.Block):
    def __init__(self, filters, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self._net = mx.gluon.nn.Sequential()
            # input size: (batch_size, 3, 64, 64)
            self._net.add(SNConv2D(filters, 4, 2, 1, 3))
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters, 32, 32)
            self._net.add(SNConv2D(filters * 2, 4, 2, 1, filters))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 2, 16, 16)
            self._net.add(SNConv2D(filters * 4, 4, 2, 1, filters * 2))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 4, 8, 8)
            self._net.add(SNConv2D(filters * 8, 4, 2, 1, filters * 4))
            self._net.add(mx.gluon.nn.BatchNorm())
            self._net.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 8, 4, 4)
            self._net.add(SNConv2D(1, 4, 1, 0, filters * 8))
            # output size: (batch_size, 1, 1, 1)

    def forward(self, images):
        y = self._net(images)
        return y.reshape((-1,))


class WassersteinLoss(mx.gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(WassersteinLoss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, fake_y, real_y=None):
        if real_y is None:
            return F.mean(-fake_y, axis=self._batch_axis, exclude=True)
        else:
            return F.mean(fake_y - real_y, axis=self._batch_axis, exclude=True)


@mx.init.register
class GANInitializer(mx.init.Initializer):
    def __init__(self, **kwargs):
        super(GANInitializer, self).__init__(**kwargs)

    def _init_weight(self, name, arr):
        if name.endswith("weight"):
            arr[:] = mx.nd.random_normal(0.0, 0.02, arr.shape)
        elif name.endswith("gamma"):
            if name.find("batchnorm") != -1:
                arr[:] = mx.nd.random_normal(1.0, 0.02, arr.shape)
            else:
                arr[:] = 1.0
        else:
            a[:] = 0.0


if __name__ == "__main__":
    net_g = Generator(64)
    net_g.initialize(GANInitializer())
    net_d = Discriminator(64)
    net_d.initialize(GANInitializer())
    loss = WassersteinLoss()
    real = mx.nd.zeros((4, 3, 64, 64))
    real_y = net_d(real)
    print("real_y: ", real_y)
    seeds = mx.nd.random_normal(shape=(4, 128, 1, 1))
    fake = net_g(seeds)
    print("fake: ", fake)
    fake_y = net_d(fake)
    print("fake_y: ", fake_y)
    print("loss_g: ", loss(fake_y))
    print("loss_d: ", loss(fake_y, real_y))
