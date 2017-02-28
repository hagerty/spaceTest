from osgeo import gdal
import os
import argparse
import sys
from bunch import Bunch
from collections import OrderedDict
import math
import cv2
import timeit
import os
import random

import tensorflow as tf
import numpy as np
import pandas as pd
#from PIL import Image

from utils import select_gpu

LAPTOP = 'MAREK_LAPTOP' in os.environ
#TRAIN_DIR = 'new-data/'
IMAGE_SIZE = (400, 400)
IMAGE_CROP = (0, 0)
NUM_CHANNELS = 0
RECTANGLES = -1
PIECES = -1
MODE_HEATMAP = 1
MODE_RECTANGLES = 2
MODE_RECTANGLES_FAKE = 3
MODE = 0
FLIP_VERTICAL = 0
FLIP_HORIZONTAL = 0

def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name='weight')

def bias_variable(shape, bias):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial, name='bias')

def conv2d(x, W, stride=1):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def myconv(signal, size, channels, filters, stddev, bias, stride=1, scope='conv'):
    with tf.variable_scope(scope):
        dev = stddev if stddev is not None else math.sqrt(1.0 / (size * size * channels))
        print 'std dev', dev
        W_conv = weight_variable([size, size, channels, filters], stddev=dev)
        b_conv = bias_variable([filters], bias=bias)
    return conv2d(signal, W_conv, stride=stride) + b_conv

def mydilatedconv(signal, size, rate, channels, filters, stddev, bias, scope='dilconv'):
    with tf.variable_scope(scope):
        dev = stddev if stddev is not None else math.sqrt(1.0 / (size * size * channels))
        print 'std dev', dev
        W_conv = weight_variable([size, size, channels, filters], stddev=dev)
        b_conv = bias_variable([filters], bias=bias)
    return tf.nn.atrous_conv2d(signal, W_conv, rate=rate, padding='SAME') + b_conv

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(signal, phase_train, scope='batch_norm', scale=True, decay=0.98):
    """
    Batch normalization
    Args:
       signal:      Tensor, 4D BHWD input maps (or any other arbitrary shape that make sense)
       phase_train: boolean, true indicates training phase, false for test time (placeholder would be a good choice)
       scope:       string, variable scope
       scale:       boolean, whether to allow output scaling
    Return:
       normed:      batch-normalized signal
    """

    with tf.variable_scope(scope):
        n_out = int(signal.get_shape()[-1])  # depth of input signal (value of the last dimension)

        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=scale)

        ema = tf.train.ExponentialMovingAverage(decay=decay)

        batch_mean, batch_variance = tf.nn.moments(signal, range(signal.get_shape().ndims - 1), name='moments')

        def mean_and_var_for_train():
            ema_apply_op = ema.apply([batch_mean, batch_variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        def mean_and_var_for_test():
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_variance)
            return ema_mean, ema_var

        mean, var = tf.cond(phase_train, mean_and_var_for_train, mean_and_var_for_test)
        normed = tf.nn.batch_normalization(signal, mean, var, beta, gamma, 1e-5)
        return normed

#returns logits
def construct_dilated(signal, input_channels, arch, stddev, bias, final_bn=True, use_batch_norm=True, phase_train=None):
    channels = input_channels

    for idx, (filters, rate) in enumerate(arch):
        prev_signal = signal
        signal = mydilatedconv(signal, size=3, rate=rate, stddev=stddev, bias=bias, channels=channels, filters=filters)
        if idx < len(arch)-1:
            if use_batch_norm:
                signal = batch_norm(signal, phase_train, scale=True)
            signal = tf.nn.relu(signal)
        elif final_bn:
            signal = batch_norm(signal, phase_train, scale=True)
        channels = filters
        print 'dil conv layer with {} filters, rate {}'.format(channels, rate)
        print signal

    return prev_signal, signal

#initially signal has shape [batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], input_channels]
def construct_convnet(signal, input_channels, arch, ending_size_dict, stddev, bias, rectangles, use_batch_norm=True, use_dropout=False, keep_prob=None, phase_train=None, ff=1):
    image_size = IMAGE_CROP
    channels = input_channels

    for idx, (filters, pool, size) in enumerate(arch):
        if use_dropout:
            print 'dropout applied'
            signal = tf.nn.dropout(signal, keep_prob)
        signal = myconv(signal, size=size, stddev=stddev, bias=bias, channels=channels, filters=filters)
        if use_batch_norm:
            signal = batch_norm(signal, phase_train, scale=True)
        signal = tf.nn.relu(signal)
        if pool:
            signal = max_pool_2x2(signal)
            image_size = ((image_size[0] + 1) / 2, (image_size[1] + 1) / 2)
        channels = filters
        print 'conv layer with {} filters, output image size {}'.format(channels, image_size)

    signal = myconv(signal, size=ending_size_dict['main'], stddev=stddev, bias=bias, channels=channels, filters=rectangles)
    print 'image_size after convolutions', image_size
    return signal, image_size


def my_read_image(s):
    s = s[:-1]
    img_path, img8_path, labels_path, heatmap_path = s.split(',')
    img = cv2.imread(img_path+'.tif').astype(float)
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.asarray(img, dtype=np.float32) / 255

    if MODE == MODE_HEATMAP:
        img8 = gdal.Open(img8_path+'.tif')
        shape = img8.GetRasterBand(1).ReadAsArray().shape
        shape += (8,)
        i8 = np.zeros(shape, dtype=np.float32)
        for i in range(1, 9):
            i8[:, :, i-1] = img8.GetRasterBand(i).ReadAsArray()
        i8 = cv2.resize(i8, IMAGE_SIZE)
        i8 = np.asarray(i8, dtype=np.float32)
        img = np.concatenate((img, i8), axis=-1)

    if heatmap_path != 'None' and heatmap_path != 'NONE':
        #print 'heatmap_path', heatmap_path
        if heatmap_path[-4:] != '.jpg':
            heatmap_path += '.jpg'
        heatmap = cv2.imread(heatmap_path).astype(float)
        heatmap = cv2.resize(heatmap, IMAGE_SIZE)
        heatmap = np.asarray(heatmap, dtype=np.float32) / 255.
    else:
        heatmap = np.zeros(IMAGE_SIZE+(3,), dtype=np.float32)

    if IMAGE_CROP != IMAGE_SIZE:
        assert(IMAGE_SIZE[0] % IMAGE_CROP[0] == 0)
        assert(IMAGE_SIZE[1] % IMAGE_CROP[1] == 0)
        dx = IMAGE_SIZE[0] / IMAGE_CROP[0]
        dy = IMAGE_SIZE[1] / IMAGE_CROP[1]
        offx = random.randrange(dx) * IMAGE_CROP[0]
        offy = random.randrange(dy) * IMAGE_CROP[1]
        heatmap = heatmap[offx:offx+IMAGE_CROP[0], offy:offy+IMAGE_CROP[1], :]
        img = img[offx:offx+IMAGE_CROP[0], offy:offy+IMAGE_CROP[1], :]

    if FLIP_VERTICAL and random.randrange(2):
        heatmap = heatmap[::-1, :, :]
        img = img[::-1, :, :]

    if FLIP_HORIZONTAL and random.randrange(2):
        heatmap = heatmap[:, ::-1, :]
        img = img[:, ::-1, :]

    if MODE != MODE_HEATMAP:
        img = np.concatenate((img, heatmap), axis=-1)

        labels = np.zeros((PIECES * PIECES * RECTANGLES), dtype=np.int32)
        if labels_path != 'NONE':
            f = open(labels_path, 'r')
            l = f.readlines()
            assert(len(l) == PIECES * PIECES)
            f.close()
            sum2 = 0
            for idx, row in enumerate(l):
                v = row.split(',')
                sum2 += int(v[0])
                labels[map(lambda s: int(s)+idx*RECTANGLES, v)[1::2]] = 1

            assert(np.sum(labels) == sum2)
    #print img.shape, labels.shape
    if MODE == MODE_HEATMAP:
        return img, heatmap
    else:
        return img, labels

def save_images(name, im):
    print 'name', name
    print 'im.shape', im.shape
    for i in range(im.shape[0]):
       cv2.imwrite('batches/'+name+'_'+str(i)+'.jpg', im[i][:, :, :3]*255)

def estimate_mean(sess, single_batch, sampled_batches=20):
    mean = None
    for batchid in range(sampled_batches):
        if MODE == MODE_HEATMAP:
            names, im, heat = sess.run(single_batch)
            save_images('im_'+str(batchid), im)
            save_images('heat_'+str(batchid), heat)
        else:
            names, im, _ = sess.run(single_batch)
            save_images('im_'+str(batchid), im)
        print 'has batch', names
        im = np.mean(im, axis=(0,1,2))
        if mean is None:
            mean = im
        else:
            mean += im
    mean /= sampled_batches
    print mean

    var = None
    for _ in range(sampled_batches):
        _, im, _ = sess.run(single_batch)
        print 'has batch'
        v = np.mean((im - mean)**2, axis=(0,1,2))
        if var is None:
            var = v
        else:
            var += v
    var /= sampled_batches
    print var
    return mean, np.sqrt(var)

def tf_batch(file_paths, batch_size, num_labels, shuffle=True):
    print file_paths[0]
    print 'file_paths_len', len(file_paths)
    print 'FLIPS', FLIP_HORIZONTAL, FLIP_VERTICAL

    if MODE == MODE_HEATMAP:
        input_queue = tf.train.slice_input_producer([file_paths], shuffle=shuffle, capacity=256)
        image, heatmap = tf.py_func(my_read_image, [input_queue[0]], [tf.float32, tf.float32])

        single_batch = tf.train.batch([input_queue[0], image, heatmap], batch_size=batch_size,
                                      num_threads=2 if LAPTOP else 6, capacity=6*batch_size,
                                      shapes=[(), (IMAGE_CROP[0], IMAGE_CROP[1], NUM_CHANNELS), (IMAGE_CROP[0], IMAGE_CROP[1], 3)])
    else:
        input_queue = tf.train.slice_input_producer([file_paths], shuffle=shuffle, capacity=256)
        image, labels = tf.py_func(my_read_image, [input_queue[0]], [tf.float32, tf.int32])

        single_batch = tf.train.batch([input_queue[0], image, labels], batch_size=batch_size,
                                      num_threads=2 if LAPTOP else 6, capacity=6*batch_size,
                                      shapes=[(), (IMAGE_SIZE[0], IMAGE_SIZE[1], 6), (num_labels,)])
    return single_batch

def read_csv():
    if MODE == MODE_RECTANGLES_FAKE:
        train_file = 'train_rect_fake.csv'
        test_file = 'test_rect.csv'
        t = np.array(open(train_file, 'r').readlines())
        valid_set = []
        train_set = []
        for idx, s in enumerate(t):
            if idx < len(t) * 4 / 5:
                train_set.append(s)
            else:
                valid_set.append(s)
        print len(valid_set), len(train_set)
    elif MODE == MODE_RECTANGLES:
        train_file = 'train_rect.csv'
        test_file = 'test_rect.csv'
        t = np.array(open(train_file, 'r').readlines())
        valid_set = []
        train_set = []
        for s in t:
            if s.find('train') != -1:
                train_set.append(s)
            else:
                valid_set.append(s)
    else:
        train_file = 'train_heatmap.csv'
        test_file = 'test_heatmap.csv'

        random.seed(123)
        t = np.array(open(train_file, 'r').readlines())
        x = range(len(t))
        random.shuffle(x)
        valid_size = len(t) / 5
        val_ids = x[:valid_size]
        train_ids = x[valid_size:]
        train_set = t[train_ids]
        valid_set = t[val_ids]

    t = np.array(open(test_file, 'r').readlines())

    print len(train_set), len(valid_set), len(t)
    print train_set[0], valid_set[0], t[0]
    print train_set[-1], valid_set[-1], t[-1]

    return train_set, valid_set, t

def dump_logits(logits, names, threshold, nameset):
    logits = logits.reshape((-1,PIECES,PIECES,RECTANGLES))
    print logits.shape, threshold
    assert(logits.shape[1] == PIECES and logits.shape[2] == PIECES)
    assert(logits.shape[3] == RECTANGLES)
    previ = -1
    a, b, c, d = np.where(logits >= threshold)
    for i, px, py, r in zip(a, b, c, d):
        if i in nameset:
            continue
        if i != previ:
            if previ >= 0:
                f.close()
            previ = i
            f = open('predictions/'+str(names[i]), 'w')
        print >>f, logits[i][px][py][r], px, py, r
    if previ != -1:
        f.close()

    #for i in range(logits.shape[0]):
    #    f = open('predictions/'+str(names[i]), 'w')
    #    for px in range(PIECES):
    #        for py in range(PIECES):
    #            for r in range(RECTANGLES):
    #                if logits[i][px][py][r] > threshold:
    #                    print >>f, logits[i][px][py][r], px, py, r
    #    f.close()

def extract_names(names):
    if names[0][-5:-1] == 'None' or names[0][-5:-1] == 'NONE':
        return map(lambda x: x.split('img')[-1][:-11], names)
    else:
        return map(lambda x: x.split('/')[-1][:-1], names)

def calc_mask(logits, labels, nonzeros):
    rect = logits.shape[1]
    assert(logits.shape == labels.shape)
    print 'logits and labels, shapes:', logits.shape, labels.shape
    print 'nonzeros', nonzeros
    res = np.array(labels)
    positions = np.argpartition(logits.reshape([-1]), -4*nonzeros)[-4*nonzeros:] #TODO constant
    print 'positions', positions, len(positions)
    a = positions / rect
    b = positions % rect
    res[a, b] = 1
    print 'mask nonzeros', np.sum(res)
    hit = np.sum(labels[a, b])
    print 'labels hit', hit, 'nonzeros', nonzeros
    return res, float(hit) / (nonzeros+0.)

def train(args):
    global PIECES, RECTANGLES, MODE, NUM_CHANNELS, IMAGE_CROP, FLIP_VERTICAL, FLIP_HORIZONTAL
    if args.augment and not args.predict:
        FLIP_VERTICAL = 1
        FLIP_HORIZONTAL = 1

    IMAGE_CROP = (args.size, args.size)
    if args.mode == 'heatmap':
        MODE = MODE_HEATMAP
        NUM_CHANNELS = 11
    elif args.mode == 'rectangles':
        MODE = MODE_RECTANGLES
        NUM_CHANNELS = 6
    else:
        assert args.mode == 'rectangles_fake'
        MODE = MODE_RECTANGLES_FAKE
        NUM_CHANNELS = 6

    PIECES = args.pieces
    RECTANGLES = args.rectangles
    device = '/cpu:0' if LAPTOP else '/gpu:{idx}'.format(idx=select_gpu())
    print 'device', device
    train_list, valid_list, test_list = read_csv()
    print len(train_list), train_list[:10], len(valid_list), valid_list[:10]
    out_len = args.pieces * args.pieces * args.rectangles

    with tf.device(device):
        print 'out_len', out_len

        if MODE != MODE_HEATMAP:
            x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS], name='x')
        else:
            x = tf.placeholder(tf.float32, shape=[None, IMAGE_CROP[0], IMAGE_CROP[1], NUM_CHANNELS], name='x')
        y_heat = tf.placeholder(tf.float32, shape=[None, IMAGE_CROP[0], IMAGE_CROP[1], 3], name='y_heat')
        y = tf.placeholder(tf.float32, shape=[None, out_len], name='y')
        mask = tf.placeholder(tf.float32, shape=[None, out_len], name='mask')
        signal = x

        phase_train = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder("float")
        learning_rate = tf.placeholder(tf.float32)
        if args.arch == 0:
            arch = [(4, 0, 3), (4, 1, 3), (8, 0, 3), (8, 1, 3), (16, 0, 3), (16, 0, 3), (16, 1, 3), (32, 0, 3)]
            end_size_dict = {'main': 3}
        elif args.arch == 1:
            arch = [(4, 0, 5), (4, 1, 5), (8, 0, 5), (8, 1, 5), (16, 0, 5), (16, 0, 5), (16, 1, 5), (32, 0, 5)]
            end_size_dict = {'main': 5}
        elif args.arch > 1:
            assert(0)

        if args.arch >= 0:
            arch = map(lambda (a,b,c): (int(args.arch_multiplier*a), b, c), arch)
            print 'arch', arch


        if args.dilarch == 0:
            dilarch = [(5, 1), (5, 1), (5, 2), (5, 4), (5, 8), (5, 16), (3, 32)]
        elif args.dilarch == 1:
            dilarch = [(10, 1), (10, 1), (10, 2), (10, 4), (10, 8), (10, 16), (3, 32)]
        elif args.dilarch == 2:
            dilarch = [(16, 1), (16, 1), (16, 2), (16, 4), (16, 8), (16, 16), (16, 32), (3, 16)]
        elif args.dilarch == 3:
            dilarch = [(16, 1), (16, 1), (16, 2), (16, 4), (16, 8), (16, 16), (16, 32), (16, 64), (3, 16)]
        else:
            assert(0)
        dilarch = map(lambda (a,b): (int(args.arch_multiplier*a), b), dilarch[:-1])+dilarch[-1:]
        print 'dilarch', dilarch


        if MODE != MODE_HEATMAP:
            if args.arch < 0:
                if args.arch == -1:
                    signal = avg_pool_2x2(signal)
                    signal = avg_pool_2x2(signal)
                    signal = myconv(signal, 8, NUM_CHANNELS, RECTANGLES, stddev=0.05, bias=0.0, stride=2)
                else:
                    assert(0)
            else:
                signal, image_size = construct_convnet(signal, input_channels=NUM_CHANNELS, arch=arch,
                                                                         ending_size_dict=end_size_dict,
                                                                         stddev=args.conv_stddev,
                                                                         bias=args.conv_bias, use_batch_norm=args.batch_norm,
                                                                         use_dropout=args.dropout > 1e-4,
                                                                         keep_prob=keep_prob, phase_train=phase_train,
                                                                         rectangles=args.rectangles)
                assert(image_size == (args.pieces, args.pieces))
            print signal
            signal = tf.reshape(signal, [-1, out_len])
            print 'final signal', signal
            #mse = tf.reduce_mean((signal - y)**2)
            #mae = tf.reduce_mean(tf.abs(signal - y))
            #small_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater(signal_small, 0.0), tf.greater(y_small, 0.5)), tf.float32))
            #small_logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(signal_small, y_small))
            logloss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(signal, y) * mask, reduction_indices=1)
            mean_logloss = tf.reduce_mean(logloss)
            objective = mean_logloss
        else:
            prev_signal, logits = construct_dilated(signal, input_channels=NUM_CHANNELS, arch=dilarch, stddev=args.conv_stddev,
                                                    bias=args.conv_bias, use_batch_norm=args.batch_norm,
                                                    final_bn=args.final_bn, phase_train=phase_train)
            softmax = tf.nn.softmax(logits)
            logloss = tf.nn.softmax_cross_entropy_with_logits(logits, y_heat)
            mean_logloss = tf.reduce_mean(logloss)
            objective = mean_logloss

        num_batches_train = len(train_list) / args.batch_size
        num_batches_valid = len(valid_list) / args.batch_size
        print 'num_batches train {}, valid {}'.format(num_batches_train, num_batches_valid)

        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(args.lr, global_step,
        #                                           num_batches_train, args.epoch_decay, staircase=False)
        if args.optimizer == 'adam':
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(objective)
        elif args.optimizer == 'momentum':
            train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(objective)
        else:
            assert(0)

        for var in tf.all_variables():
            print var.name

        train_batch = tf_batch(train_list, batch_size=args.batch_size, num_labels=out_len)
        valid_batch = tf_batch(valid_list, batch_size=args.batch_size, num_labels=out_len)
        test_batch = tf_batch(test_list, batch_size=args.batch_size, num_labels=out_len)

        saver = tf.train.Saver(max_to_keep=10)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            print 'session started'
            sess.run(tf.initialize_all_variables())
            print 'variables initialized'
            threads = tf.train.start_queue_runners(sess, coord=tf.train.Coordinator())
            print 'queues started'
            mean, stddev = estimate_mean(sess, train_batch, sampled_batches=5)
            print 'mean estimated, shape={}, flattened={}'.format(mean.shape, mean)

            epoch = 0
            if args.restore_path:
                saver.restore(sess, args.restore_path)

            if args.predict:
                pass_list = [(train_batch, len(train_list), 'train'),
                             (valid_batch, len(valid_list), 'val'),
                             (test_batch, len(test_list), 'test')]
                #pass_list = [(train_batch, 10, 'train'), (valid_batch, 10, 'val'), (test_batch, 10, 'test')]
                for (batch_gen, size, pref) in pass_list:
                    print size
                    nameset = {}
                    while len(nameset) < size:
                        print len(nameset)
                        if MODE == MODE_HEATMAP:
                            names, batch_x, _ = sess.run(batch_gen)
                            batch_x -= mean
                            batch_x /= stddev
                            names = map(lambda s: pref+s, extract_names(names))
                            print 'names', names[:10]
                            #TODO prev_signal read here
                            features, heat_pred = sess.run([prev_signal, softmax], feed_dict={x: batch_x, phase_train: False}) #TODO check false
                            print 'features.shape', features.shape
                            for i in range(args.batch_size):
                                if names[i] not in nameset:
                                    nameset[names[i]] = 1
                                    cv2.imwrite('predictions/'+names[i]+'.jpg', np.asarray(heat_pred[i] * 255, dtype=int))
                                    f = open('predictions/'+names[i]+'-numpy-tab', 'w')
                                    np.save(f, features[i])
                                    f.close()
                        else:
                            names, batch_x, _ = sess.run(batch_gen)
                            batch_x -= mean
                            batch_x /= stddev
                            [logits] = sess.run([signal], feed_dict={x: batch_x, phase_train: False, keep_prob: 1.0-args.dropout})
                            names = extract_names(names)
                            print 'names', names[:10]
                            print 'logits mean', np.mean(logits)
                            cnt = args.batch_size * args.pred_per_image
                            threshold = np.partition(logits.reshape([-1]), -cnt)[-cnt]
                            dump_logits(logits, names, threshold, nameset)
                            for i in range(args.batch_size):
                                if names[i] not in nameset:
                                    nameset[names[i]] = 1

                return


            epoch_lr = args.lr
            while epoch < args.num_epochs:
                epoch += 1
                print 'epoch {} learning rate {}'.format(epoch, epoch_lr)
                epoch_start_time = timeit.default_timer()
                train_time, load_time, valid_time, dump_time = 0, 0, 0, 0

                if MODE == MODE_HEATMAP:
                    pass_list = [(([train_step, mean_logloss] if args.train else [mean_logloss]), num_batches_train, train_batch, True),
                                 ([mean_logloss], num_batches_valid, valid_batch, False)]
                    for compute_list, num_batches, batch_gen, pt in pass_list:
                        load_time = 0
                        main_time = 0
                        pass_start = timeit.default_timer()
                        results = []
                        for i in range(num_batches):
                            start = timeit.default_timer()
                            _, batch_x, batch_heat = sess.run(batch_gen)
                            batch_x -= mean
                            batch_x /= stddev
                            load_time += timeit.default_timer() - start
                            main_time -= timeit.default_timer()
                            res = sess.run(compute_list, feed_dict={x: batch_x, y_heat: batch_heat, phase_train: pt, learning_rate: epoch_lr})
                            main_time += timeit.default_timer()
                            results.append(res)
                            if i < 20000:
                                print 'train', i, res, np.mean(np.array(results, dtype=float), axis=0)
                        full_time = timeit.default_timer() - pass_start
                        rest_time = full_time - load_time - main_time
                        pass_res = np.mean(np.array(results, dtype=float), axis=0)
                        print 'Pass res {} time: {} main_time: {} load_time: {} rest: {}'.format(pass_res, full_time, main_time, load_time, rest_time)
                else:
                    pass_list = [(([train_step, mean_logloss] if args.train else [mean_logloss]), num_batches_train, train_batch, True),
                     ([mean_logloss], num_batches_valid, valid_batch, False)]
                    for compute_list, num_batches, batch_gen, pt in pass_list:
                        load_time = 0
                        main_time = 0
                        pass_start = timeit.default_timer()
                        results = []
                        for i in range(num_batches):
                            start = timeit.default_timer()
                            _, batch_x, batch_y = sess.run(batch_gen)
                            batch_x -= mean
                            batch_x /= stddev
                            load_time += timeit.default_timer() - start
                            main_time -= timeit.default_timer()
                            [logits] = sess.run([signal], feed_dict={x: batch_x, phase_train: pt, keep_prob: 1.0-args.dropout})
                            print 'logits mean', np.mean(logits)
                            nonzeros = np.sum(batch_y)
                            batch_mask, train_acc = calc_mask(logits, batch_y, nonzeros)
                            res = sess.run(compute_list, feed_dict={x: batch_x, y: batch_y, mask: batch_mask, phase_train: pt, learning_rate: epoch_lr})
                            res.append(train_acc)
                            main_time += timeit.default_timer()
                            results.append(res)
                            if i < 20000:
                                print 'train', i, res, np.mean(np.array(results, dtype=float), axis=0)
                        full_time = timeit.default_timer() - pass_start
                        rest_time = full_time - load_time - main_time
                        pass_res = np.mean(np.array(results, dtype=float), axis=0)
                        print 'Pass res {} time: {} main_time: {} load_time: {} rest: {}'.format(pass_res, full_time, main_time, load_time, rest_time)

                save_path = saver.save(sess, 'models/epoch{}.ckpt'.format(epoch))
                print ('Model saved in file: %s' % save_path)

                epoch_lr *= args.epoch_decay
                epoch_time = timeit.default_timer() - epoch_start_time
                rest_time = epoch_time - train_time - load_time - valid_time
                print 'Epoch {} time {}, train {}, valid {}, load {}, dumping {}, rest {}'.\
                    format(epoch, epoch_time, train_time, valid_time, load_time, dump_time, rest_time)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--ff', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--conv_stddev', type=float, default=None)
    parser.add_argument('--conv_bias', type=float, default=0.01)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--final_bn', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fc_stddev', type=float, default=0.01)
    parser.add_argument('--fc_bias', type=float, default=0.01)
    parser.add_argument('--l2_reg_fc', type=float, default=0.0001)
    parser.add_argument('--epoch_decay', type=float, default=0.95)
    parser.add_argument('--predict', type=int, default=0)
    parser.add_argument('--pred_per_image', type=int, default=1000)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--rectangles', type=int, required=True)
    parser.add_argument('--pieces', type=int, required=True)
    parser.add_argument('--arch', type=int, default=0)
    parser.add_argument('--dilarch', type=int, default=0)
    parser.add_argument('--arch_multiplier', type=float, default=1.)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--size', type=int, default=400)
    parser.add_argument('--augment', type=int, default=0)
    parser.add_argument('--train', type=int, default=1)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    print args
    train(args)
