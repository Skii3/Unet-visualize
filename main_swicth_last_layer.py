#! /usr/bin/python
# -*- coding: utf8 -*-

from model import *
from utils import *
import scipy.io as sio
from config import configPara
import scipy
import functools
import os
# construct net
graph = tf.Graph()
with graph.as_default():
    nn1 = configPara.nn1
    input1 = tf.placeholder('float32', [1, nn1, nn1, 1], name='input_image_forward1')
    target1 = tf.placeholder('float32', [1, nn1, nn1, 1], name='target_image_forward1')
    forward1 = generator(input1, reuse=False, net_name="net_forward")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph)
    sess.run(tf.global_variables_initializer())

    # read weights
    checkpoint = tf.train.latest_checkpoint("train_result/checkpoint/")
    print checkpoint
    tf.train.Saver().restore(sess, checkpoint)

    filters = [op.name for op in graph.get_operations() if "filter" in op.name[-6:]]
    print(len(filters))

    train_vars = tf.trainable_variables()
    vars_forward = [var for var in train_vars if 'net_forward' in var.name]
    vars_forward_name = [var.name for var in train_vars if 'net_forward' in var.name]

    weight_list = []
    for i in range(0, len(vars_forward), 1):
        temp = sess.run(vars_forward[i])
        weight_list.append(temp)

# without residual
graph2 = tf.Graph()
with graph2.as_default():
    nn1 = configPara.nn1
    input2 = tf.placeholder('float32', [1, nn1, nn1, 1], name='input2_image_forward1')
    target2 = tf.placeholder('float32', [1, nn1, nn1, 1], name='target2_image_forward1')
    forward2_temp = generator(input2, reuse=False, net_name="net_forward")
    forward2 = input2 - forward2_temp

    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph2)
    sess2.run(tf.global_variables_initializer())

    # read weights
    checkpoint = tf.train.latest_checkpoint("train_result_without_residual/checkpoint/")
    print checkpoint
    tf.train.Saver().restore(sess2, checkpoint)

    filters = [op.name for op in graph.get_operations() if "filter" in op.name[-6:]]
    print(len(filters))

    train_vars2 = tf.trainable_variables()
    vars_forward2 = [var for var in train_vars2 if 'net_forward' in var.name]
    vars_forward2_name = [var.name for var in train_vars2 if 'net_forward' in var.name]

    weight_list2 = []
    for i in range(0, len(vars_forward2), 1):
        temp = sess2.run(vars_forward2[i])
        weight_list2.append(temp)

    for i in range(0, len(vars_forward_name), 1):
        if "output" in vars_forward_name[i]:
            temp = graph2.get_tensor_by_name(vars_forward2_name[i])
            temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[i]), weight_list[i])
            sess2.run(temp)

# sample result of this model
train_imgs = read_all_imgs(configPara.train_image_path, regx='.*.txt')
train_imgs = expand_imgs(train_imgs)

dataNum = len(train_imgs)
seq = random.randint(0, dataNum - 1)
imgs_input = train_imgs[0]
mean_input = np.mean(imgs_input)
print(mean_input)
imgs_input1, imgs_target1 = sampleImg(imgs_input, configPara.nn1)

out2 = sess2.run(forward2, {input2: imgs_input1})
scipy.misc.imsave('./denoise_switch.bmp', out2.squeeze())
scipy.misc.imsave('./noisedata.bmp', imgs_input1.squeeze())
scipy.misc.imsave('./clean.bmp', imgs_target1.squeeze())
