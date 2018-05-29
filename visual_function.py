
#! /usr/bin/python
# -*- coding: utf8 -*-

from model import *
from utils import *
import scipy.io as sio
from config import configPara
import scipy
import functools
import os

nn1 = 128
img_noise = np.zeros([1, nn1, nn1, 1])

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    with graph3.as_default():
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective

        t_grad = tf.gradients(t_score, input1)[0]

        img = img0.copy()
        for i in range(iter_n):

            g, score = sess.run([t_grad, t_score], {input1: img})
            # normalizing the gradient, so the same step size should work
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step
            if i % 100 == 0:
                print(i, ' ', score, ' \n')
            if iter_n <= 100 and i % 10 == 0 or iter_n == 2:
                print(i, ' ', score, ' \n')
    return img

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph3.get_tensor_by_name("%s:0" % layer)

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * np.eye(1, dtype=np.float32)

def lap_split(img):
    '''split the img into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1, 2, 2, 1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5 * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img - lo2
    return lo, hi

def lap_split_n(img, n):
    '''build laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return  levels[::-1]

def lap_merge(levels):
    '''merge laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5 * 4, tf.shape(hi), [1, 2, 2, 1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''perform the laplacian pyramid normalization'''
    #img = tf.expand_dims(img, 0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0, :, :, :]


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def render_lapnorm(t_obj, img0=img_noise, iter_n=10, step=1.0, lap_n=4):
    with graph3.as_default():
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, input1)[0]  # behold the power of automatic differentiation!
        #funcTemp = functools.partial(lap_normalize, scale_n=lap_n)
        #lap_norm_func = tffunc(np.float32)(funcTemp)
        g_input = tf.placeholder('float32', [1, nn1, nn1, 1])
        gTemp = lap_normalize(g_input)
        img = img0.copy()
        for i in range(iter_n):

            g, score = sess.run([t_grad, t_score], {input1: img})
            g = sess.run(gTemp, feed_dict={g_input:g})

            img += g * step
            if i % 100 == 0:
                print(i, ' ', score, ' \n')
    return img

def render_tv(t_obj, img0=img_noise, iter_n=10, step=1.0, lam=1000):
    with graph3.as_default():
        tvDiff_loss_forward = tf.reduce_mean(
            tf.image.total_variation(input1)) / (nn1 * nn1) / 1
        reduce_sum = tf.reduce_mean(t_obj)
        t_score = reduce_sum - tvDiff_loss_forward * lam
        t_grad = tf.gradients(t_score, input1)[0]  # behold the power of automatic differentiation!
        #funcTemp = functools.partial(lap_normalize, scale_n=lap_n)
        #lap_norm_func = tffunc(np.float32)(funcTemp)
        g_input = tf.placeholder('float32', [1, nn1, nn1, 1])
        gTemp = lap_normalize(g_input)
        img = img0.copy()
        for i in range(iter_n):

            g, score, sum, tv = sess.run([t_grad, t_score, reduce_sum, tvDiff_loss_forward], {input1: img})

            #g = sess.run(gTemp, feed_dict={g_input:g})
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step * 0.001
            if i % 100 == 0:
                print(i, ' ', score, ' ', sum, ' ', tv*lam, ' \n')
            if iter_n <= 100 and i % 10 == 0 or iter_n == 2:
                print(i, ' ', score, ' ', sum, ' ', tv*lam, ' \n')
    return img
