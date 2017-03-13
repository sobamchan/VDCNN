import numpy as np
import sobamchan_chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import chainer


class Conv_Block(sobamchan_chainer.Model):

    def __init__(self, in_channels=None, out_channels=256, ksize=(1, 3)):
        super(Conv_Block, self).__init__(
            conv1=L.Convolution2D(in_channels, out_channels, ksize),
            conv2=L.Convolution2D(out_channels, out_channels, ksize),
            bnorm1=L.BatchNormalization(out_channels),
            bnorm2=L.BatchNormalization(out_channels),
        )

    def __call__(self, x, train=True):
        return self.fwd(x, train)

    def fwd(self, x, train=True):
        h = F.relu(self.bnorm1(self.conv1(x)))
        h = F.relu(self.bnorm2(self.conv2(h)))
        return h


class Conv_Blocks(sobamchan_chainer.Model):

    def __init__(self, block_n, in_channels, out_channels, ksize=(1, 3)):
        super(Conv_Blocks, self).__init__()
        modules = []
        for i in range(block_n):
            modules += [('conv_block_{}'.format(i), Conv_Block(in_channels, out_channels, ksize))]
            in_channels = out_channels
        [ self.add_link(*link) for link in modules ]
        self.modules = modules
        self.block_n = block_n

    def __call__(self, x, train=True):
        return self.fwd(x, train)

    def fwd(self, x, train=True):
        for i in range(self.block_n):
            x = self['conv_block_{}'.format(i)](x, train)
        return x


class VDCNN(sobamchan_chainer.Model):

    def __init__(self, vocab_n, embedding_size, class_n, in_channels=1, out_channels=64, ksize=(1, 3), block_sizes={}):
        super(VDCNN, self).__init__()
        modules = []
        modules += [ ('embed', L.EmbedID(vocab_n, embedding_size)) ]
        modules += [ ('conv', L.Convolution2D(in_channels, 64, ksize)) ]
        if block_sizes == {}:
            block_sizes[64] = 2
            block_sizes[128] = 2
            block_sizes[256] = 2
            block_sizes[512] = 2
        for feature_n, block_n in block_sizes.items():
            modules += [('conv_blocks_{}'.format(feature_n), Conv_Blocks(block_n, None, feature_n))]
        modules += [ ('fc1', L.Linear(None, 2048)) ]
        modules += [ ('fc2', L.Linear(None, 2048)) ]
        modules += [ ('fc3', L.Linear(None, class_n)) ]
        [ self.add_link(*link) for link in modules ]

        self.block_sizes = block_sizes
        self.modules = modules
        self.class_n = class_n

    def __call__(self, x, t, train=True):
        y = self.fwd(x, train)
        return F.softmax_cross_entropy(y, t)

    def fwd(self, x, train=True):
        h = self['embed'](x)
        h = F.reshape(h, (h.shape[0], 1, h.shape[3], h.shape[2]))
        h = self['conv'](h)
        for feature_n, block_n in self.block_sizes.items():
            h = self['conv_blocks_{}'.format(feature_n)](h)
            if feature_n != 512:
                h = F.max_pooling_2d(h, (1, 2))
        h = F.max_pooling_2d(h, (1, 5))     #TODO change to 'k max pooling'
        h = F.relu(self['fc1'](h))
        h = F.relu(self['fc2'](h))
        h = self['fc3'](h)
        return h
