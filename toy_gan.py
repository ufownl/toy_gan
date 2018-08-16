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
    def __init__(self, filters, feature_size=128, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        with self.name_scope():
            self._encoder = mx.gluon.nn.Sequential()
            self._decoder = mx.gluon.nn.Sequential()
            # input size: (batch_size, 3, 64, 64)
            self._encoder.add(mx.gluon.nn.Conv2D(filters, 4, 2, 1, use_bias=False))
            self._encoder.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters, 32, 32)
            self._encoder.add(mx.gluon.nn.Conv2D(filters * 2, 4, 2, 1, use_bias=False))
            self._encoder.add(mx.gluon.nn.BatchNorm())
            self._encoder.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 2, 16, 16)
            self._encoder.add(mx.gluon.nn.Conv2D(filters * 4, 4, 2, 1, use_bias=False))
            self._encoder.add(mx.gluon.nn.BatchNorm())
            self._encoder.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 4, 8, 8)
            self._encoder.add(mx.gluon.nn.Conv2D(filters * 8, 4, 2, 1, use_bias=False))
            self._encoder.add(mx.gluon.nn.BatchNorm())
            self._encoder.add(mx.gluon.nn.LeakyReLU(0.2))
            # state size: (batch_size, filters * 8, 4, 4)
            self._encoder.add(mx.gluon.nn.Conv2D(feature_size, 4, 1, 0, use_bias=False))
            # feature size: (batch_size, feature_size, 1, 1)
            self._decoder.add(mx.gluon.nn.Conv2DTranspose(filters * 8, 4, 1, 0,  use_bias=False))
            self._decoder.add(mx.gluon.nn.BatchNorm())
            self._decoder.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters * 8, 4, 4)
            self._decoder.add(mx.gluon.nn.Conv2DTranspose(filters * 4, 4, 2, 1, use_bias=False))
            self._decoder.add(mx.gluon.nn.BatchNorm())
            self._decoder.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters * 4, 8, 8)
            self._decoder.add(mx.gluon.nn.Conv2DTranspose(filters * 2, 4, 2, 1, use_bias=False))
            self._decoder.add(mx.gluon.nn.BatchNorm())
            self._decoder.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters * 2, 16, 16)
            self._decoder.add(mx.gluon.nn.Conv2DTranspose(filters, 4, 2, 1, use_bias=False))
            self._decoder.add(mx.gluon.nn.BatchNorm())
            self._decoder.add(mx.gluon.nn.Activation("relu"))
            # state size: (batch_size, filters, 32, 32)
            self._decoder.add(mx.gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False))
            self._decoder.add(mx.gluon.nn.Activation("tanh"))
            # output size: (batch_size, 3, 64, 64)

    def forward(self, images):
        features = self._encoder(images)
        energy = self._decoder(features) - images
        return _l2_norm(energy.reshape((0, -1)), axis=1), features.reshape((0, -1))


def pull_away_term(features):
    normalized = features / _l2_norm(features, axis=1, keepdims=True)
    similarity = mx.nd.dot(normalized, normalized, transpose_b=True)
    batch_size = features.shape[0]
    return (mx.nd.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))

def _l2_norm(x, axis=0, keepdims=False):
    return mx.nd.sqrt(mx.nd.sum(mx.nd.square(x), axis=axis, keepdims=keepdims))


if __name__ == "__main__":
    net_g = Generator(64)
    net_g.initialize(mx.init.Xavier())
    net_d = Discriminator(64)
    net_d.initialize(mx.init.Xavier())
    real = mx.nd.ones((4, 3, 64, 64))
    real_y, real_f = net_d(real)
    print("real_y: ", real_y)
    print("real_f: ", real_f)
    print("real_pt: ", pull_away_term(real_f))
    seeds = mx.nd.random_normal(shape=(4, 128, 1, 1))
    fake = net_g(seeds)
    print("fake: ", fake)
    fake_y, fake_f = net_d(fake)
    print("fake_y: ", fake_y)
    print("fake_f: ", fake_f)
    print("fake_pt: ", pull_away_term(fake_f))
