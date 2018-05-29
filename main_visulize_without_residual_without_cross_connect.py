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

    weight_list = []
    for i in range(0, len(vars_forward), 1):
        temp = sess.run(vars_forward[i])
        weight_list.append(temp)

sio.savemat('weight_all.mat', {'weights': weight_list})

# sample result of this model
train_imgs = read_all_imgs(configPara.train_image_path, regx='.*.txt')
train_imgs = expand_imgs(train_imgs)

dataNum = len(train_imgs)
seq = random.randint(0, dataNum - 1)
imgs_input = train_imgs[0]
mean_input = np.mean(imgs_input)
print(mean_input)
imgs_input1, imgs_target1 = sampleImg(imgs_input, configPara.nn1)


'''     original model      '''
net_layers = [op.name for op in graph.get_operations() if (op.type == 'Conv2D'
                                                           or ( "conv2d_transpose" in op.name and "output_shape"
                                                                not in op.name))
              and "net_forward" in op.name or "input_image_forward1" in op.name ]\
             #or ("lrelu" in op.name and "add" in op.name)or "batchnorm/add_1" in op.name or "Relu" in op.name]
net_feature_map = []
for i in range(0, len(net_layers), 1):
    tensor = graph.get_tensor_by_name(net_layers[i]+':0')
    temp = sess.run(tensor, feed_dict={input1: imgs_input1})
    str = net_layers[i]
    str = str.replace("/", "_")
    file_name = configPara.samples_save_dir + '/' + str
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    for j in range(0, temp.shape[-1], 1):
        temp2 = temp[:, :, :, j]
        scipy.misc.imsave(file_name + '/channel_%d.png' % j, temp2.squeeze())

out = sess.run(forward1, {input1: imgs_input1, target1: imgs_target1})
scipy.misc.imsave(configPara.samples_save_dir + '/denoise.bmp', out.squeeze())
scipy.misc.imsave(configPara.samples_save_dir + '/noisedata.bmp', imgs_input1.squeeze())
scipy.misc.imsave(configPara.samples_save_dir + '/clean.bmp', imgs_target1.squeeze())

graph3 = tf.Graph()

graph3 = graph



# visualize first layer
img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input


visual_type = 1


layers = [op.name for op in graph3.get_operations() if op.type == 'Conv2D' and "net_forward" in op.name]
if visual_type == 2:
    layers = [op.name for op in graph3.get_operations() if "conv2d_transpose" in op.name and "output_shape" not in op.name]
#layers = [op.name for op in graph3.get_operations() if "conv2d_transpose" in op.name and "output_shape" not in op.name]
print layers
feature_nums = [int(graph3.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
print feature_nums


if visual_type == -1:    # c1,c2,c3,output,d1,d2,d3
    # visualize channels in different layers
    ind = 0
    layer = layers[ind]
    #img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input
    img_noise = np.zeros([1, nn1, nn1, 1]) + mean_input
    index1 = (ind + 1) * 16
    index2 = (ind + 1) * 16
    iter = 2000
    feature_num = feature_nums[ind]
    channel = np.round(feature_num/2)
    img = render_naive(T(layer)[:, :, :, channel], img_noise, iter)
    #img = render_tv(T(layer)[:, :, :, channel], img_noise, iter, 1.0, 5000)
    str = layer
    str = str.replace("/", "_")
    scipy.misc.imsave('0visual_output_feature_d%d/%s_%d_%d_%d_iter%d_tv.png' % (ind+1, str, channel, index1, index2, iter),
                      img.squeeze())
    out = sess.run(forward1, {input1: img})
    scipy.misc.imsave(
        '0visual_output_feature_d%d/out_%s_%d_%d_%d_iter%d_tv.png' % (ind + 1, str, channel, index1, index2, iter),
        out.squeeze())

    out = sess.run(forward1, {input1: imgs_input1, target1: imgs_target1})

    scipy.misc.imsave('0visual_output_feature_d1/denoise.png', out.squeeze())
    scipy.misc.imsave('0visual_output_feature_d1/noisedata.png', imgs_input1.squeeze())
    scipy.misc.imsave('0visual_output_feature_d1/clean.png', imgs_target1.squeeze())


if visual_type == 0:    # c1,c2,c3,output,d1,d2,d3
    # visualize channels in different layers
    layer = layers[0]
    img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input
    [a,b,c,d] = graph3.get_tensor_by_name(layers[0] + ':0').get_shape()
    for channel in range(0, d, 1):
        for index1 in range(3, b, int(b/4)):
            for index2 in range(3, c, int(c/4)):
                iter = 2000
                img = render_naive(T(layer)[:, index1, index2, channel], img_noise, iter)
                # img = render_tv(T(layer)[:, index1, index2, channel], img_noise, iter, 1.0, 100)
                # img = render_lapnorm(T(layer)[:, index1, index2, channel], img_noise, iter)
                str = layer
                str = str.replace("/", "_")
                file_name = './naive_feature_d1'
                if not os.path.exists(file_name):
                    os.mkdir(file_name)
                scipy.misc.imsave(file_name+'/%s_%d_%d_%d_iter%d.png' % (str, channel, index1, index2, iter), img.squeeze())
elif visual_type == 1:      # c1,c2,c3
    # visualize every channel with sliced shape in each layer

    for ind in range(2,4,1):
        img_all = []
        layer = layers[ind]
        feature_num = feature_nums[ind]
        img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input
        #img_noise = np.zeros([1, nn1, nn1, 1]) + mean_input
        for channel in range(0, feature_num, 1):
            index1 = 8 * (3-ind)
            index2 = 8 * (3-ind)
            iter = 0
            if ind == 3:
                index1 = 30
                index2 = 30
                iter = 1000

            if ind <= 3:
                img = render_naive(T(layer)[:, index1, index2, channel], img_noise, iter)
            #else:
            #    img = render_tv(T(layer)[:, index1, index2, channel], img_noise, iter, 1.0, 5000)
            #img2 = render_naive(T(layer)[:, index1, index2, channel], img_noise, iter)
            if ind == 0:
                img_save = img[:, 2 * index1 - 1:2 * index1 - 1 + 4, 2 * index2 - 1:2 * index2 - 1 + 4, :]
            elif ind == 1:
                img_save = img[:, 2*(2*index1-1)-1:2*(2*index1-1+3)-1+4, 2*(2*index2-1)-1:2*(2*index2-1+3)-1+4, :]
            elif ind == 2:
                img_save = img[:, 2 * (2 * (2 * index1 - 1) - 1) - 1: 2 * (2 * (2 * index1 - 1 + 3) - 1 + 3) - 1 + 4,
                           2 * (2 * (2 * index2 - 1) - 1) - 1: 2 * (2 * (2 * index2 - 1 + 3) - 1 + 3) - 1 + 4, :]
                #img_save2 = img2[:, 2 * (2 * (2 * index1 - 1) - 1) - 1: 2 * (2 * (2 * index1 - 1 + 3) - 1 + 3) - 1 + 4,
                #           2 * (2 * (2 * index2 - 1) - 1) - 1: 2 * (2 * (2 * index2 - 1 + 3) - 1 + 3) - 1 + 4, :]
            elif ind == 3:
                d1 = index1 - 1
                d2 = index1 + 3
                c1 = np.int(np.ceil((d1 + 1) / 2) - 1)
                c2 = np.int(np.ceil((d2 + 1) / 2) - 1 + 1)
                b1 = np.int(np.ceil((c1 + 1) / 2) - 1)
                b2 = np.int(np.ceil((c2 + 1) / 2) - 1 + 1)
                a1 = np.int(np.ceil((b1 + 1) / 2))
                a2 = np.int(np.ceil((b2 + 1) / 2))

                d11 = index2 - 1
                d22 = index2 + 3
                c11 = np.int(np.ceil((d11 + 1) / 2) - 1)
                c22 = np.int(np.ceil((d22 + 1) / 2) - 1 + 1)
                b11 = np.int(np.ceil((c11 + 1) / 2) - 1)
                b22 = np.int(np.ceil((c22 + 1) / 2) - 1 + 1)
                a11 = np.int(np.ceil((b11 + 1) / 2))
                a22 = np.int(np.ceil((b22 + 1) / 2))
                img_save = img[:, 8 * a1 - 15:8 * a2 + 14 + 1, 8 * a11 - 15:8 * a22 + 14 + 1, :]
            if ind < 3:
                file_name = "naive_feature_all_c%d" % (ind + 1)
            else:
                file_name = "naive_feature_all_output"
            if not os.path.exists(file_name):
                os.mkdir(file_name)
            str = layer
            str = str.replace("/", "_")
            scipy.misc.imsave(
                file_name + '/%s_%d_%d_%d_iter%d.png' % ( str, channel, index1, index2, iter),
                img_save.squeeze())

            img_all.append(img_save.squeeze())
            #scipy.misc.imsave(
            #    'naive_feature_all_c%d_lap/%s_%d_%d_%d_iter%d.png' % (ind + 1, str, channel, index1, index2, iter),
            #    img_save2.squeeze())
        sio.savemat(file_name + '.mat', {'img_all': img_all})


elif visual_type == 2:      # d1,d2,d3
    # visualize every channel with sliced shape in each layer
    for ind in range(1,3,1):
        layer = layers[ind]
        feature_num = feature_nums[ind]
        img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input
        #img_noise = np.zeros([1, nn1, nn1, 1]) + mean_input
        img_all = []
        iter = 3000
        for channel in range(0, feature_num, 1):
            index1 = (ind + 1) * 16
            index2 = (ind + 1) * 16

            if ind == 1:
                index1 = 10
                index2 = 10
            elif ind == 2:
                index1 = 65
                index2 = 65


            #img = render_naive(T(layer)[:, index1, index2, channel], img_noise, iter)
            img = render_tv(T(layer)[:, index1, index2, channel], img_noise, iter, 1.0, 100)
            #img = render_lapnorm(T(layer)[:, index1, index2, channel], img_noise, iter)
            if ind == 0:
                a = np.int(np.ceil((index1 + 1) / 2))
                b = np.int(np.ceil((index2 + 1) / 2))
                img_save = img[:, 8*a-15:8*a+14+1, 8*b-15:8*b+14+1, :]
            elif ind == 1:
                b1 = np.int(np.ceil((index1 + 1) / 2) - 1)
                b2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
                a1 = np.int(np.ceil((b1 + 1) / 2))
                a2 = np.int(np.ceil((b2 + 1) / 2))
                b11 = np.int(np.ceil((index2 + 1) / 2) - 1)
                b22 = np.int(np.ceil((index2 + 1) / 2) - 1 + 1)
                a11 = np.int(np.ceil((b11 + 1) / 2))
                a22 = np.int(np.ceil((b22 + 1) / 2))
                img_save = img[:, 8 * a1 - 15:8 * a2 + 14 + 1, 8 * a11 - 15:8 * a22 + 14 + 1, :]
            elif ind == 2:
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
                img_save = img[:, 8 * a1 - 15:8 * a2 + 14 + 1, 8 * a11 - 15:8 * a22 + 14 + 1, :]

            file_name = "naive_feature_all_d%d"% (ind + 1)
            if not os.path.exists(file_name):
                os.mkdir(file_name)

            str = layer
            str = str.replace("/", "_")
            scipy.misc.imsave(
                file_name+'/%s_%d_%d_%d_iter_tv%d.png' % (str, channel, index1, index2, iter),
                img_save.squeeze())
            img_all.append(img_save.squeeze())
        sio.savemat('naive_feature_all_d%d_iter%d.mat' % (ind + 1,iter), {'img_all': img_all})
