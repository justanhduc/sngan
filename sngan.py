import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import theano
from theano import tensor as T
import time
from collections import OrderedDict

import neuralnet as nn
from neuralnet import read_data

srng = theano.sandbox.rng_mrg.MRG_RandomStreams(np.random.randint(1, int(time.time())))


class DCGANGenerator(nn.Sequential):
    def __init__(self, input_shape, bottom_width=4, ch=512, wscale=0.02, hidden_activation='relu',
                 output_activation='tanh', layer_name='DCGANGen'):
        super(DCGANGenerator, self).__init__(input_shape=input_shape, layer_name=layer_name)
        self.append(nn.FullyConnectedLayer(self.output_shape, ch * bottom_width ** 2, layer_name=layer_name + '/fc1',
                                           activation='linear', init=nn.Normal(wscale)))
        self.append(nn.BatchNormLayer(self.output_shape, layer_name+'/bn_fc1', activation=hidden_activation, epsilon=2e-5))
        self.append(nn.ReshapingLayer(self.output_shape, (-1, ch, bottom_width, bottom_width), layer_name+'/reshape'))

        shape = [o * 2 for o in self.output_shape[2:]]
        self.append(
            nn.TransposedConvolutionalLayer(self.output_shape, ch // 2, 4, shape, layer_name=layer_name + '/deconv1',
                                            padding=1, activation='linear', init=nn.Normal(wscale)))
        self.append(nn.BatchNormLayer(self.output_shape, layer_name+'/bn_deconv1', activation=hidden_activation, epsilon=2e-5))

        shape = [o * 2 for o in self.output_shape[2:]]
        self.append(
            nn.TransposedConvolutionalLayer(self.output_shape, ch // 4, 4, shape, layer_name=layer_name + '/deconv2',
                                            padding=1, activation='linear', init=nn.Normal(wscale)))
        self.append(nn.BatchNormLayer(self.output_shape, layer_name + '/bn_deconv2', activation=hidden_activation, epsilon=2e-5))

        shape = [o * 2 for o in self.output_shape[2:]]
        self.append(
            nn.TransposedConvolutionalLayer(self.output_shape, ch // 8, 4, shape, layer_name=layer_name + '/deconv3',
                                            padding=1, activation='linear', init=nn.Normal(wscale)))
        self.append(nn.BatchNormLayer(self.output_shape, layer_name + '/bn_deconv3', activation=hidden_activation, epsilon=2e-5))

        self.append(
            nn.TransposedConvolutionalLayer(self.output_shape, 3, 3, layer_name=layer_name + '/output', stride=(1, 1),
                                            activation=output_activation, init=nn.Normal(wscale)))


class SNDCGANDiscriminator(nn.Sequential):
    def __init__(self, input_shape, ch=512, wscale=0.02, output_dim=1, layer_name='SNDCGANDis'):
        super(SNDCGANDiscriminator, self).__init__(input_shape=input_shape, layer_name=layer_name)
        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch // 8, 3, nn.Normal(wscale), False, stride=1, activation='lrelu',
                                  layer_name=layer_name + '/conv1', alpha=.2))
        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch // 4, 4, nn.Normal(wscale), False, stride=2, border_mode=1,
                                  activation='lrelu', layer_name=layer_name + '/conv2', alpha=.2))

        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch // 4, 3, nn.Normal(wscale), False, stride=1, activation='lrelu',
                                  layer_name=layer_name + '/conv3', alpha=.2))
        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch // 2, 4, nn.Normal(wscale), False, stride=2, border_mode=1,
                                  activation='lrelu', layer_name=layer_name + '/conv4', alpha=.2))

        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch // 2, 3, nn.Normal(wscale), False, stride=1, activation='lrelu',
                                  layer_name=layer_name + '/conv5', alpha=.2))
        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch, 4, nn.Normal(wscale), False, stride=2, border_mode=1,
                                  activation='lrelu', layer_name=layer_name + '/conv6', alpha=.2))
        self.append(
            nn.ConvolutionalLayer(self.output_shape, ch, 3, nn.Normal(wscale), False, stride=1, activation='lrelu',
                                  layer_name=layer_name + '/conv7', alpha=.2))

        self.append(nn.FullyConnectedLayer(self.output_shape, output_dim, nn.Normal(wscale), activation='linear',
                                           layer_name=layer_name + '/output'))


def _l2normalize(v, eps=1e-12):
    return v / ((T.sum(v ** 2)) ** 0.5 + eps)


def max_singular_value(W, u=None, lp=1):
    """
    Apply power iteration for the weight parameter
    """
    if W.ndim > 2:
        W = W.flatten(2)

    if u is None:
        u = theano.shared(np.random.normal(size=(1, W.get_value().shape[0])).astype('float32'), 'u')
    _u = u
    for _ in range(lp):
        _v = _l2normalize(T.dot(_u, W))
        _u = _l2normalize(T.dot(_v, W.T))
    sigma = T.sum(T.dot(T.dot(_u, W), _v.T))
    return sigma, _u, _v


def spectral_normalize(updates):
    new_updates = OrderedDict()
    for key in updates:
        param = updates[key]
        if param.ndim < 2 or any(a in key.name for a in ('grad', 'beta', 'gamma')):
            new_updates[key] = param
        else:
            u = theano.shared(np.random.normal(size=(1, key.get_value().shape[0])).astype('float32'), key.name + '/u')
            sigma, _u, _ = max_singular_value(param, u)
            new_updates[key] = param / (sigma + 1e-12)
            new_updates[u] = _u
    return new_updates


class DataManager(nn.DataManager):
    def __init__(self, placeholders, n_iters, batchsize, shuffle):
        super(DataManager, self).__init__(None, placeholders, path='D:\DB\cifar10', batch_size=batchsize,
                                          n_epochs=n_iters, shuffle=shuffle)
        self.load_data()
        self.n_epochs = int(self.batch_size * n_iters / self.data_size)

    def load_data(self):
        X_train, _, X_test, _ = read_data.load_dataset(self.path)
        X = np.concatenate((X_train, X_test))
        self.dataset = np.float32(self.normalize(X / 255.))
        self.data_size = self.dataset.shape[0]

    def normalize(self, input):
        return (input - .5) * 2.

    def unnormalize(self, input=None):
        return input / 2. + .5


def train_sngan(z_dim=128, image_shape=(3, 32, 32), bs=64, n_iters=int(1e5), n_dis=5):
    gen = DCGANGenerator((None, z_dim))
    dis = SNDCGANDiscriminator(gen.output_shape)

    z = srng.uniform((bs, z_dim), -1, 1, ndim=2, dtype='float32')
    X = T.tensor4('image', 'float32')
    X_ = theano.shared(np.zeros((bs,) + image_shape, 'float32'), 'image_placeholder')

    # training
    nn.set_training_status(True)
    X_fake = gen(z)
    y_fake = dis(X_fake)
    y_real = dis(X)

    dis_loss_real = T.mean(T.nnet.softplus(-y_real))
    dis_loss_fake = T.mean(T.nnet.softplus(y_fake))
    dis_loss = dis_loss_real + dis_loss_fake
    gen_loss = T.mean(T.nnet.softplus(-y_fake))

    updates_gen = nn.adam(gen_loss, gen.trainable, 2e-4, 0., .9)
    updates_dis = nn.adam(dis_loss, dis.trainable, 2e-4, 0., .9)
    updates_dis_norm = spectral_normalize(updates_dis)

    train_gen = nn.function([], gen_loss, updates=updates_gen)
    train_dis = nn.function([], dis_loss, updates=updates_dis_norm, givens={X: X_})

    # testing
    nn.set_training_status(False)
    fixed_noise = T.constant(np.random.uniform(-1, 1, (bs, z_dim)), 'fixed noise', 2, 'float32')
    gen_imgs = gen(fixed_noise)
    generate = nn.function([], gen_imgs, name='generate images')

    dm = DataManager(X_, n_iters, bs, True)
    mon = nn.monitor.Monitor(model_name='LSGAN', use_visdom=True)
    epoch = 0
    print('Training...')
    batches = dm.get_batches(epoch, dm.n_epochs, infinite=True)
    start = time.time()
    for iteration in range(n_iters):
        #update generator
        training_gen_cost = train_gen()
        if np.isnan(training_gen_cost) or np.isinf(training_gen_cost):
            raise ValueError('Training failed due to NaN cost')
        mon.plot('training gen cost', training_gen_cost)

        #update discriminator
        training_disc_cost = []
        for i in range(n_dis):
            batches.__next__()
            training_disc_cost.append(train_dis())
            if np.isnan(training_disc_cost[-1]) or np.isinf(training_disc_cost[-1]):
                raise ValueError('Training failed due to NaN cost')
        mon.plot('training disc cost', np.mean(training_disc_cost))

        if iteration % 500 == 0:
            gen_images = generate()
            mon.imwrite('generated image', dm.unnormalize(gen_images))
            mon.plot('time elapsed', (time.time() - start)/60.)
            mon.flush()
        mon.tick()
    mon.flush()
    print('Training finished!')


if __name__ == '__main__':
    train_sngan()
