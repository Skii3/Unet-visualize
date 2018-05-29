#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
from model import *
from utils import *
import PIL
from config import configPara
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def train():
    loss_file = open("loss.txt", 'a')
    print(configPara)
    print("---------------------------reading image---------------------------")
    test_imgs = read_all_imgs(configPara.test_image_path, regx='.*.txt')
    target_imgs = read_all_imgs(configPara.test_image_path_target, regx='.*.txt')


    train_imgs = read_all_imgs(configPara.train_image_path, regx='.*.txt')
    train_imgs = expand_imgs(train_imgs)

    print("---------------------------defining model---------------------------")

    nn1 = configPara.nn1
    input1 = tf.placeholder('float32', [1, nn1, nn1, 1], name='input_image_forward1')
    target1 = tf.placeholder('float32', [1, nn1, nn1, 1], name='target_image_forward1')
    forward1 = generator(input1, reuse=False, net_name="net_forward")
    # result1=generator(forward1, reuse=True,net_name="net_forward")
    # target_forward1=generator(target1, reuse=True,net_name="net_forward")

    pixel_num = nn1 * nn1
    L1_loss_forward = tf.reduce_mean(tf.abs(forward1 - target1))

    temp_snr = tf.reduce_sum(tf.square(tf.abs(target1))) /\
               tf.reduce_sum(tf.square(tf.abs(target1-forward1)))
    snr_output = 10.0 * tf.log(temp_snr) / tf.log(10.0)

    temp_snr2 = tf.reduce_sum(tf.square(tf.abs(target1))) / \
               tf.reduce_sum(tf.square(tf.abs(target1 - input1)))
    snr_input = 10.0 * tf.log(temp_snr2) / tf.log(10.0)

    diff = forward1 - target1
    # L1_loss_forward=tf.reduce_mean(tf.multiply(diff,diff))* configPara.L1_lambda
    # tvDiff_loss_forward=tf.reduce_mean(tf.image.total_variation(forward1-target1)) /pixel_num * configPara.tvDiff_lambda
    tvDiff_loss_forward = tf.reduce_mean(
        tf.image.total_variation(forward1)) / pixel_num * configPara.tvDiff_lambda / 10000
    # loop_loss=tf.reduce_mean(tf.abs(generator(input1-forward1, reuse=True,net_name="net_forward")))
    loss_forward = L1_loss_forward + tvDiff_loss_forward
    # loss_forward=L1_loss_forward + tvDiff_loss_forward
    # loss_forward=tvDiff_loss_forward
    # loss_forward = L1_loss_forward
    # loss_forward=tf.nn.l2_loss(forward1-target1)

    train_vars = tf.trainable_variables()
    vars_forward = [var for var in train_vars if 'net_forward' in var.name]
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(configPara.lr_init, trainable=False)

    optim_forward = tf.train.AdamOptimizer(lr_v, beta1=configPara.beta1).minimize(loss_forward, var_list=vars_forward)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=vars_forward, max_to_keep=100)

    print("---------------------------restoring model---------------------------")
    if configPara.if_continue_train:
        checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
        print '580+'
        print checkpoint
        tf.train.Saver().restore(sess, checkpoint)

    print("---------------------------training model---------------------------")
    dataNum = len(train_imgs)
    n_epoch = configPara.n_epoch
    for epoch in range(0, n_epoch + 1):
        epoch_time = time.time()
        sum_all_loss, sum_tvDiff_loss, sum_L1_loss = 0, 0, 0
        n_iter = 0
        snr_input_max = 0
        snr_output_max = 0
        snr_input_min = 0
        snr_output_min = 0
        for idx in range(0, 1000, 1):
            step_time = time.time()
            seq = random.randint(0, dataNum - 1)
            imgs_input = train_imgs[seq]
            imgs_input1, imgs_target1 = sampleImg(imgs_input, configPara.nn1)
            # scipy.misc.imsave(configPara.buffer_dir+'/trainImg%d%d.png'%(epoch,idx),imgs_input1.squeeze())
            if np.size(imgs_input1) == 1:
                continue
            tvDiff_loss, L1_loss, _, snr_input1, snr_output1 \
                = sess.run([tvDiff_loss_forward, L1_loss_forward, optim_forward, snr_input, snr_output], \
                           {input1: imgs_input1, target1: imgs_target1})

            if idx == 0:
                snr_input_max = snr_input1
                snr_input_min = snr_input1
            else:
                if snr_input_max < snr_input1:
                    snr_input_max = snr_input1
                    snr_output_max = snr_output1
                if snr_input_min > snr_input1:
                    snr_input_min = snr_input1
                    snr_output_min = snr_output1
            sum_all_loss = sum_all_loss + tvDiff_loss + L1_loss
            sum_tvDiff_loss = sum_tvDiff_loss + tvDiff_loss
            sum_L1_loss = sum_L1_loss + L1_loss
            n_iter += 1

        loss_content = str(epoch) + ' ' + str(sum_L1_loss / n_iter) + ' ' + str(sum_all_loss / n_iter) + '\n'
        loss_file.write(loss_content)

        print("[*] Epoch [%2d/%2d] %4d time: %4.4fs, all_loss: %.8f, tvDiff_loss: %.8f, L1_loss: %.8f, "
              "max_snr: %.4f => %.4f, min_snr: %.4f => %.4f" \
              % (epoch, n_epoch, n_iter, time.time() - epoch_time, sum_all_loss / n_iter, sum_tvDiff_loss / n_iter,
                 sum_L1_loss / n_iter,snr_input_max,snr_output_max,snr_input_min,snr_output_min))

        if (epoch % configPara.save_model_freq == 0):
            tf.train.Saver().save(sess, configPara.checkpoint_dir + '/model%d' % epoch)

            seq = random.randint(0, dataNum - 1)
            imgs_input = train_imgs[seq]
            imgs_input, imgs_target = sampleImg(imgs_input, configPara.nn1)
            if np.size(imgs_input) == 1:
                continue
            out = sess.run(forward1, {input1: imgs_input})
            result = np.concatenate((imgs_input.squeeze(), out.squeeze(),  imgs_target.squeeze()), axis=1)

            for i in range(9):
                seq = random.randint(0, dataNum - 1)
                imgs_input = train_imgs[seq]
                imgs_input, imgs_target = sampleImg(imgs_input, configPara.nn1)
                if np.size(imgs_input) == 1:
                    continue
                out = sess.run(forward1, {input1: imgs_input})
                out = np.concatenate((imgs_input.squeeze(), out.squeeze(), imgs_target.squeeze()), axis=1)
                result = np.concatenate((result, out), axis=0)

            scipy.misc.imsave(configPara.samples_save_dir + '/train_forward%d.png' % epoch, result)

        if epoch % configPara.test_freq == 0:
            if configPara.type != 0:
                print 'error test file path!!!'
            print("---------------------------test--------------------------------")
            runTest(test_imgs, target_imgs, sess)

def runTest(test_imgs, target_imgs, sess):
    #if configPara.test:
    #    checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
    #    tf.train.Saver().restore(sess, checkpoint)
    inputImage = test_imgs[0]
    w, h = inputImage.shape

    n = 8
    h2 = h - h % n
    w2 = w - w % n

    input_image_forward_large = tf.placeholder('float32', [1, w2, h2, 1], name='input_test')
    image_forward_large = generator(input_image_forward_large, reuse=True, net_name="net_forward")
    input_snr_sum = 0
    output_snr_sum = 0
    for i in range(0, len(test_imgs), 1):
        inputImage = test_imgs[i]
        targetImage = target_imgs[i]
        w, h = inputImage.shape

        n = 8
        h2 = h - h % n
        w2 = w - w % n

        inputImage2 = inputImage[0:w2, 0:h2]
        targetImage2 = targetImage[0:w2, 0:h2]
        # test_img = test_img[0:w2,0:h2]
        inputImage2 = np.expand_dims(np.expand_dims(inputImage2, axis=0), axis=3)
        out = sess.run(image_forward_large, {input_image_forward_large: inputImage2})

        tmp_snr0 = np.sum(np.square(np.abs(targetImage2.squeeze()))) / np.sum(
            np.square(np.abs(targetImage2.squeeze() - inputImage2.squeeze())))
        input_snr = 10.0 * np.log(tmp_snr0) / np.log(10.0)  # 输入图片的snr

        tmp_snr0 = np.sum(np.square(np.abs(targetImage2.squeeze()))) / np.sum(
            np.square(np.abs(targetImage2.squeeze() - out.squeeze())))
        output_snr = 10.0 * np.log(tmp_snr0) / np.log(10.0)  # 输出图片的snr

        input_snr_sum = input_snr_sum + input_snr
        output_snr_sum = output_snr_sum + output_snr
        # np.save(configPara.test_save_dir+'/my_%d.txt' % i,out.squeeze())

        np.savetxt(configPara.test_save_dir + '/%d_denoise.txt' % i, out.squeeze(), delimiter=' ')
        np.savetxt(configPara.test_save_dir + '/%d_noise.txt' % i, inputImage2.squeeze(), delimiter=' ')
        # np.savetxt(configPara.test_save_dir+'/target_%d.txt' % i, test_img, delimiter=' ')
        # out = scipy.misc.imresize(reScale(out.squeeze()), size=[w, h], interp='bicubic', mode=None)
        # out=np.concatenate((inputImage, out),axis=1)
        scipy.misc.imsave(configPara.test_save_dir + '/%d_denoise.png' % i, reScale(out.squeeze()))
        scipy.misc.imsave(configPara.test_save_dir + '/%d_noise.png' % i, reScale(inputImage2.squeeze()))
    input_snr_sum = input_snr_sum / len(test_imgs)
    output_snr_sum = output_snr_sum / len(test_imgs)
    print('[*]input snr average: %.2f, output snr average: %.2f' % (input_snr_sum, output_snr_sum))


def runTest_test(test_imgs, target_imgs,sess):
    for i in range(0, len(test_imgs), 1):
        inputImage = test_imgs[i]
        targetImage = target_imgs[i]
        w, h = inputImage.shape

        n = 8
        h2 = h - h % n
        w2 = w - w % n
        if i == 0:
            input_image_forward_large = tf.placeholder('float32', [1, w2, h2, 1], name='input_test')
            image_forward_large = generator(input_image_forward_large, reuse=False, net_name="net_forward")

            if configPara.test:
                checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
                print checkpoint
                tf.train.Saver().restore(sess, checkpoint)

        inputImage2 = inputImage[0:w2, 0:h2]
        targetImage2 = targetImage[0:w2, 0:h2]
        #mean1 = np.mean(inputImage2)
        #mean2 = np.mean(targetImage2)
        #std1 = np.std(inputImage2)
        #std2 = np.std(targetImage2)
        #inputImage2 = (inputImage2 - mean1) / std1
        #targetImage2 = (targetImage2 - mean2) / std2
        # test_img = test_img[0:w2,0:h2]
        inputImage2 = np.expand_dims(np.expand_dims(inputImage2, axis=0), axis=3)
        out = sess.run(image_forward_large, {input_image_forward_large: inputImage2})

        # np.save(configPara.test_save_dir+'/my_%d.txt' % i,out.squeeze())
        #inputImage2 = inputImage2.squeeze() * std1 + mean1
        #out = out.squeeze() * std1 + mean1
        #targetImage2 = targetImage2.squeeze() * std2 + mean2

        tmp_snr0 = np.sum(np.square(np.abs(targetImage2.squeeze()))) / np.sum(np.square(np.abs(targetImage2.squeeze() - inputImage2.squeeze())))
        input_snr = 10.0 * np.log(tmp_snr0) / np.log(10.0)  # 输入图片的snr

        tmp_snr0 = np.sum(np.square(np.abs(targetImage2.squeeze()))) / np.sum(np.square(np.abs(targetImage2.squeeze() - out.squeeze())))
        output_snr = 10.0 * np.log(tmp_snr0) / np.log(10.0)  # 输出图片的snr
        print('[*]input snr: %.2f, output snr: %.2f' % (input_snr,output_snr))

        np.savetxt(configPara.test_save_dir + '/%d_denoise.txt' % round(input_snr), out.squeeze(), delimiter=' ')
        np.savetxt(configPara.test_save_dir + '/%d_noise.txt' % round(input_snr), inputImage2.squeeze(), delimiter=' ')
        # np.savetxt(configPara.test_save_dir+'/target_%d.txt' % i, test_img, delimiter=' ')
        # out = scipy.misc.imresize(reScale(out.squeeze()), size=[w, h], interp='bicubic', mode=None)
        # out=np.concatenate((inputImage, out),axis=1)
        scipy.misc.imsave(configPara.test_save_dir + '/%d_denoise.png' % round(input_snr), reScale(out.squeeze()))
        scipy.misc.imsave(configPara.test_save_dir + '/%d_noise.png' % round(input_snr), reScale(inputImage2.squeeze()))

def test():
    print(configPara)
    test_imgs = read_all_imgs(configPara.test_image_path, regx='.*.txt')
    target_imgs = read_all_imgs(configPara.test_image_path_target, regx='.*.txt')
    # test_imgs = generate_noiseimgs(test_img,20)

    # input_image_forward_large = tf.placeholder('float32', [1, 512, 512, 1], name='input_test')
    # image_forward_large = generator(input_image_forward_large, reuse=False, net_name="net_forward")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    #sess.run(tf.global_variables_initializer())
    # checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
    # tf.train.Saver().restore(sess,checkpoint)

    runTest_test(test_imgs, target_imgs, sess)

def test_20():
    print(configPara)
    file_list = os.listdir(configPara.test_image_path_20)
    files = [f for f in file_list if f[0] != '.']
    tag = 0
    file_all = []
    for file in os.listdir(configPara.test_image_path_20+files[0]):
        if file[0] == '.':
            continue
        file_all.append(file)

    file_all = sorted(file_all, key=lambda x: int(x))

    for file in file_all:
        test_imgs = read_all_imgs(configPara.test_image_path_20+files[0]+'/'+file+'/', regx='.*.txt')
        target_imgs = read_all_imgs(configPara.test_image_path_20+files[1]+'/'+file+'/', regx='.*.txt')
        # test_imgs = generate_noiseimgs(test_img,20)
        input_snr = 0
        output_snr = 0
        for i in range(20):
            inputImage = test_imgs[i]
            targetImage = target_imgs[i]
            w, h = inputImage.shape

            n = 8
            h2 = h - h % n
            w2 = w - w % n
            if tag == 0:
                input_image_forward_large = tf.placeholder('float32', [1, w2, h2, 1], name='input_test')
                image_forward_large = generator(input_image_forward_large, reuse=False, net_name="net_forward")
                sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
                if configPara.test:
                    checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
                    print checkpoint
                    tf.train.Saver().restore(sess, checkpoint)
                tag = 1

            inputImage2 = inputImage[0:w2, 0:h2]
            targetImage2 = targetImage[0:w2, 0:h2]
            # test_img = test_img[0:w2,0:h2]
            inputImage2 = np.expand_dims(np.expand_dims(inputImage2, axis=0), axis=3)
            out = sess.run(image_forward_large, {input_image_forward_large: inputImage2})

            # np.save(configPara.test_save_dir+'/my_%d.txt' % i,out.squeeze())

            tmp_snr0 = np.sum(np.square(np.abs(targetImage2.squeeze()))) / np.sum(
                np.square(np.abs(targetImage2.squeeze() - inputImage2.squeeze())))
            input_snr = input_snr + 10.0 * np.log(tmp_snr0) / np.log(10.0)  # 输入图片的snr

            tmp_snr0 = np.sum(np.square(np.abs(targetImage2.squeeze()))) / np.sum(
                np.square(np.abs(targetImage2.squeeze() - out.squeeze())))
            output_snr = output_snr + 10.0 * np.log(tmp_snr0) / np.log(10.0)  # 输出图片的snr

        print('[*]input snr: %.2f, output snr: %.2f' % (input_snr/20, output_snr/20))


if configPara.type == 0:
    train()
    os.system('cp ./console_log.txt ./train_result_without_aug/train_log.txt')
else:
    if configPara.type == 1:
        test()          #测试pluto测试集、sigsbee
        os.system('cp ./console_log.txt ./train_result_without_aug/test_plut.txt')
    if configPara.type == 2:
        test()
        os.system('cp ./console_log.txt ./train_result_without_aug/test_sigsbee.txt')
    if configPara.type == 3:
        test_20()      #测试20次sigsbee
        os.system('cp ./console_log.txt ./train_result_without_aug/test_sigsbee20.txt')


