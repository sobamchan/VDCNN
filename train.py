import numpy as np
from chainer import optimizers
import fire
from tqdm import tqdm
import sys

from model import VDCNN

from sobamchan_iterator import Iterator
from sobamchan_dataset import AGCorpus
from token_dict.basic import token_dict
import sobamchan_utility
import sobamchan_slack
s = sobamchan_slack.Slack
util = sobamchan_utility.Utility()


def train(epoch=10, batch_size=128, embedding_size=16, class_n=10):
    test_ratio = .2
    
    # fake dataset
    # vocab_n = 100
    # X = np.random.randint(vocab_n, size=(1000, 1, 1024)).astype(np.int32)
    # T = np.random.randint(10, size=(1000)).astype(np.int32)
    # train_x, test_x = X[:int(len(X)*(1-test_ratio))], X[-int(len(X)*test_ratio):]
    # train_t, test_t = T[:int(len(T)*(1-test_ratio))], T[-int(len(T)*test_ratio):]

    maxlen = 1014
    vocab_n = len(token_dict)
    ag = AGCorpus('./datas/newsspace200.xml')
    T, X = ag.get_data()
    N = len(X)
    X = util.np_int32([ util.convert_one_of_m_vector_char(x, token_dict, maxlen).astype(np.int32).reshape(1, maxlen) for x in X ])
    T = util.np_int32(T)
    train_x, test_x = X[:int(len(X)*(1-test_ratio))], X[-int(len(X)*test_ratio):]
    train_t, test_t = T[:int(len(T)*(1-test_ratio))], T[-int(len(T)*test_ratio):]

    train_n = len(train_x)
    test_n = len(test_x)

    model = VDCNN(len(token_dict), embedding_size, class_n)
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(model)

    s.s_print('epoch: {}'.format(epoch))
    s.s_print('batch size: {}'.format(batch_size))
    s.s_print('embedding size: {}'.format(embedding_size))
    s.s_print('class n: {}'.format(class_n))
    s.s_print('vocab n: {}'.format(vocab_n))
    s.s_print('train n: {}'.format(train_n))
    s.s_print('test n: {}'.format(test_n))

    for e in range(epoch):
        loss_acc = 0
        order = np.random.permutation(train_n)
        train_iter_x = Iterator(train_x, batch_size, order=order)
        train_iter_t = Iterator(train_t, batch_size, order=order)
        for x, t in tqdm(zip(train_iter_x, train_iter_t)):
            x = model.prepare_input(x, np.int32)
            t = model.prepare_input(t, np.int32)
            loss = model(x, t)
            loss.backward()
            optimizer.update()
            loss_acc += float(loss.data)
        print('loss: {}'.format(loss_acc/train_n/batch_size))
        order = np.random.permutation(train_n)
        test_iter_x = Iterator(test_x, batch_size, order=order)
        test_iter_t = Iterator(test_t, batch_size, order=order)
        for x, t in tqdm(zip(test_iter_x, test_iter_t)):
            x = model.prepare_input(x, np.int32)
            t = model.prepare_input(t, np.int32)
            loss = model(x, t)
            loss_acc += float(loss.data)
        print('test loss: {}'.format(loss_acc/test_n/batch_size))


if __name__ == '__main__':
    fire.Fire()
