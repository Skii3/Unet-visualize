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
    checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
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

sio.savemat('weight_all.mat', {'weights': weight_list})

visual_target = 1 # -1:c3 0:d1, 1:d2, 2:d3
graph2 = tf.Graph()
if visual_target == -1:
    with graph2.as_default():
        print('visualize c3 for output')
        c3_0 = tf.placeholder('float32', [1, 16, 16, 512], name='c3_0')
        ngf = 64
        ind = 0
        with tf.variable_scope("net2_forward"):
            c3 = batchnorm(c3_0, "b3")
            d1 = batchnorm(deconv(tf.nn.relu(c3), ngf * 4, "d1"), "b1_2")
            d2 = batchnorm(deconv(tf.nn.relu(d1), ngf * 2, "d2"), "b2_2")
            d3 = deconv(tf.nn.relu(d2), ngf, "d3")
            output = tf.tanh(conv(tf.nn.relu(d3), 1, "output", 1))

        sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph2)
        sess2.run(tf.global_variables_initializer())
        train_vars2 = tf.trainable_variables()
        vars_forward2_name = [var.name for var in train_vars2 if 'net2_forward' in var.name]
        for i in range(0, len(vars_forward),1):
            if "b3" in vars_forward_name[i]:
                offset = i
                break
        for i in range(offset, len(vars_forward_name), 1):
            j = i - offset
            if "d2" in vars_forward2_name[j] or "d3" in vars_forward2_name[j]:
                weight_temp = weight_list[i]
                weight_temp2 = weight_temp[:, :, :, 0:np.int(np.shape(weight_temp)[-1] / 2)]
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_temp2)
                sess2.run(temp)
            else:
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_list[i])
                sess2.run(temp)
        temp_c3 = np.zeros(shape=[1,16,16,512])
        index1 = 8
        index2 = 8
        file_name="c3_output"
        img_list = []
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        for channel in range(0,512,1):
            temp_c3[0, index1, index2, channel] = 1000
            output_eval = sess2.run(output, {c3_0:temp_c3})
            img_save = output_eval[:, 2 * (2 * (2 * index1 - 1) - 1) - 1: 2 * (2 * (2 * index1 - 1 + 3) - 1 + 3) - 1 + 4,
                           2 * (2 * (2 * index2 - 1) - 1) - 1: 2 * (2 * (2 * index2 - 1 + 3) - 1 + 3) - 1 + 4, :]
            scipy.misc.imsave(file_name+'/output_%d_%d_%d.png' % (channel,index1,index2), img_save.squeeze())
            img_list.append(img_save)

        sio.savemat(file_name + '.mat', {'img_list': img_list})
elif visual_target == 0:
    with graph2.as_default():
        print('visualize d1 for output')
        d1_0 = tf.placeholder('float32', [1, 32, 32, 256], name='d1_0')
        ngf = 64
        ind = 0
        with tf.variable_scope("net2_forward"):
            d1 = batchnorm(d1_0, "b1_2")
            d2 = batchnorm(deconv(tf.nn.relu(d1), ngf * 2, "d2"), "b2_2")
            d3 = deconv(tf.nn.relu(d2), ngf, "d3")
            output = tf.tanh(conv(tf.nn.relu(d3), 1, "output", 1))

        sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph2)
        sess2.run(tf.global_variables_initializer())
        train_vars2 = tf.trainable_variables()
        vars_forward2_name = [var.name for var in train_vars2 if 'net2_forward' in var.name]
        for i in range(0, len(vars_forward),1):
            if "b1_2" in vars_forward_name[i]:
                offset = i
                break
        for i in range(offset, len(vars_forward_name), 1):
            j = i - offset
            if "d2" in vars_forward2_name[j] or "d3" in vars_forward2_name[j]:
                weight_temp = weight_list[i]
                weight_temp2 = weight_temp[:, :, :, 0:np.int(np.shape(weight_temp)[-1] / 2)]
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_temp2)
                sess2.run(temp)
            else:
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_list[i])
                sess2.run(temp)
        temp_d1 = np.zeros(shape=[1, 32, 32, 256])
        index1 = 16
        index2 = 16
        file_name="d1_output"
        img_list = []
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        for channel in range(0,256,1):
            temp_d1[0, index1, index2, channel] = 1000
            output_eval = sess2.run(output, {d1_0:temp_d1})
            a = np.int(np.ceil((index1 + 1) / 2))
            b = np.int(np.ceil((index2 + 1) / 2))
            img_save = output_eval[:, 8 * a - 15+9:8 * a + 14 + 1-6, 8 * b - 15+9:8 * b + 14 + 1-6, :]
            scipy.misc.imsave(file_name+'/output_%d_%d_%d.png' % (channel,index1,index2), img_save.squeeze())
            img_list.append(img_save)

        sio.savemat(file_name + '.mat', {'img_list': img_list})
elif visual_target == 1:
    with graph2.as_default():
        print('visualize d2 for output')
        d2_0 = tf.placeholder('float32', [1, 64, 64, 128], name='d2_0')
        ngf = 64
        ind = 0
        with tf.variable_scope("net2_forward"):
            d2 = batchnorm(d2_0, "b2_2")
            d3 = deconv(tf.nn.relu(d2), ngf, "d3")
            output = tf.tanh(conv(tf.nn.relu(d3), 1, "output", 1))

        sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph2)
        sess2.run(tf.global_variables_initializer())
        train_vars2 = tf.trainable_variables()
        vars_forward2_name = [var.name for var in train_vars2 if 'net2_forward' in var.name]
        for i in range(0, len(vars_forward),1):
            if "b2_2" in vars_forward_name[i]:
                offset = i
                break
        for i in range(offset, len(vars_forward_name), 1):
            j = i - offset
            if "d2" in vars_forward2_name[j] or "d3" in vars_forward2_name[j]:
                weight_temp = weight_list[i]
                weight_temp2 = weight_temp[:, :, :, 0:np.int(np.shape(weight_temp)[-1] / 2)]
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_temp2)
                sess2.run(temp)
            else:
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_list[i])
                sess2.run(temp)
        temp_d1 = np.zeros(shape=[1, 64, 64, 128])
        index1 = 30
        index2 = 30
        file_name="d2_output"
        img_list = []
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        for channel in range(0,128,1):
            temp_d1[0, index1, index2, channel] = 1000
            output_eval = sess2.run(output, {d2_0:temp_d1})
            b1 = np.int(np.ceil((index1 + 1) / 2) - 1)
            b2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
            a1 = np.int(np.ceil((b1 + 1) / 2))
            a2 = np.int(np.ceil((b2 + 1) / 2))
            b11 = np.int(np.ceil((index2 + 1) / 2) - 1)
            b22 = np.int(np.ceil((index2 + 1) / 2) - 1 + 1)
            a11 = np.int(np.ceil((b11 + 1) / 2))
            a22 = np.int(np.ceil((b22 + 1) / 2))
            img_save = output_eval[:, 8 * a1 - 15+15:8 * a2 + 14 + 1-15, 8 * a11 - 15+15:8 * a22 + 14 + 1-15, :]
            scipy.misc.imsave(file_name+'/output_%d_%d_%d.png' % (channel,index1,index2), img_save.squeeze())
            img_list.append(img_save)

        sio.savemat(file_name + '.mat', {'img_list': img_list})

elif visual_target == 2:
    with graph2.as_default():
        print('visualize d3 for output')
        d3_0 = tf.placeholder('float32', [1, 128, 128, 64], name='d3_0')
        ngf = 64
        ind = 0
        with tf.variable_scope("net2_forward"):
            output = tf.tanh(conv(tf.nn.relu(d3_0), 1, "output", 1))

        sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph2)
        sess2.run(tf.global_variables_initializer())
        train_vars2 = tf.trainable_variables()
        vars_forward2_name = [var.name for var in train_vars2 if 'net2_forward' in var.name]
        for i in range(0, len(vars_forward),1):
            if "output" in vars_forward_name[i]:
                offset = i
                break
        for i in range(offset, len(vars_forward_name), 1):
            j = i - offset
            if "d2" in vars_forward2_name[j] or "d3" in vars_forward2_name[j]:
                weight_temp = weight_list[i]
                weight_temp2 = weight_temp[:, :, :, 0:np.int(np.shape(weight_temp)[-1] / 2)]
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_temp2)
                sess2.run(temp)
            else:
                temp = graph2.get_tensor_by_name(vars_forward2_name[j])
                temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[j]), weight_list[i])
                sess2.run(temp)
        temp_d1 = np.zeros(shape=[1, 128, 128, 64])
        index1 = 64
        index2 = 64
        file_name="d3_output"
        img_list = []
        if not os.path.exists(file_name):
            os.mkdir(file_name)
        for channel in range(0,64,1):
            temp_d1[0, index1, index2, channel] = 1000
            output_eval = sess2.run(output, {d3_0:temp_d1})
            c1 = np.int(np.ceil((index1 + 1) / 2) - 1)
            c2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
            b1 = np.int(np.ceil((c1 + 1) / 2) - 1)
            b2 = np.int(np.ceil((c2 + 1) / 2) - 1 + 1)
            a1 = np.int(np.ceil((b1 + 1) / 2))
            a2 = np.int(np.ceil((b2 + 1) / 2))
            c11 = np.int(np.ceil((index2 + 1) / 2) - 1)
            c22 = np.int(np.ceil((index2 + 1) / 2) - 1 + 1)
            b11 = np.int(np.ceil((c11 + 1) / 2) - 1)
            b22 = np.int(np.ceil((c22 + 1) / 2) - 1 + 1)
            a11 = np.int(np.ceil((b11 + 1) / 2))
            a22 = np.int(np.ceil((b22 + 1) / 2))
            img_save = output_eval[:, 8 * a1 - 15+15:8 * a2 + 14 + 1-15, 8 * a11 - 15+15:8 * a22 + 14 + 1-15, :]
            scipy.misc.imsave(file_name+'/output_%d_%d_%d.png' % (channel,index1,index2), img_save.squeeze())
            img_list.append(img_save)

        sio.savemat(file_name + '.mat', {'img_list': img_list})