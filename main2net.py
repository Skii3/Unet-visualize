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
    loss_file = open("loss1e-3_11_15.txt", 'a')
    print(configPara)
    print("---------------------------reading image---------------------------")
    test_imgs = read_all_imgs(configPara.test_image_path, regx='.*.txt')
    #test_imgs = generate_noiseimgs(test_img,20)
    train_imgs = read_all_imgs(configPara.train_image_path, regx='.*.txt')
    train_imgs = expand_imgs(train_imgs)

    #train_imgs = read_all_imgs(configPara.train_image_path, regx='.*.txt')
    #train_imgs2 = read_all_imgs(configPara.train_image_path2, regx='.*.txt')

    print("---------------------------defining model---------------------------")

    nn1=configPara.nn1
    nn2=configPara.nn2
    input1 = tf.placeholder('float32', [1, nn1, nn1, 1], name='input_image_forward1')
    target1 = tf.placeholder('float32', [1, nn1, nn1, 1], name='target_image_forward1')
    forward1 = generator(input1, reuse=False,net_name="net_forward")

    input2 = tf.placeholder('float32', [1, nn2, nn2, 1], name='input_image_forward2')
    target2 = tf.placeholder('float32', [1, nn2, nn2, 1], name='target_image_forward2')
    forward2 = gammaGenerator(input2, reuse=False,net_name="gammaGenerator")
    #gamma = tf.clip_by_value(gamma,0,1,name="clip")
    #forward2 = tf.multiply(input1,gamma) + tf.multiply(forward1,1-gamma)


    loss_forward1 = tf.reduce_mean(tf.abs(forward1-target1)) \
                    + 0.25*tf.reduce_mean(gammaGenerator(forward1-input1, reuse=True,net_name="gammaGenerator")) \
                    + 0.25*tf.reduce_mean(gammaGenerator(input1-forward1, reuse=True,net_name="gammaGenerator"))
    #loss_forward1 = tf.nn.l2_loss(forward1-target1)
    #loss_forward2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=forward2, labels=target2))
    #loss_forward2 = tf.sqrt(tf.reduce_mean(tf.square(forward2 - target2)))
    loss_forward2 = tf.reduce_mean(tf.abs(forward2-target2))

    train_vars = tf.trainable_variables()
    vars_forward1 = [var for var in train_vars if 'net_forward' in var.name]
    optim_forward1= tf.train.AdamOptimizer(2e-4, beta1=configPara.beta1).minimize(loss_forward1, var_list=vars_forward1)

    vars_forward2 = [var for var in train_vars if 'gammaGenerator' in var.name]
    optim_forward2= tf.train.AdamOptimizer(2e-4, beta1=configPara.beta1).minimize(loss_forward2, var_list=vars_forward2)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    #saver = tf.train.Saver(var_list=vars_forward, max_to_keep=100)
    print("---------------------------restoring model---------------------------")
    if configPara.if_continue_train:
        checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
        tf.train.Saver().restore(sess,checkpoint)

    print("---------------------------training model---------------------------")
    dataNum=len(train_imgs)
    n_epoch=configPara.n_epoch
    
    for epoch in range(1, 26):
        seq=random.randint(0, dataNum-1)
        imgs_input=train_imgs[seq]
        imgs_input2,imgs_target2=sampleImg2(imgs_input,configPara.nn1)
        result=np.concatenate((imgs_input2.squeeze(), imgs_target2.squeeze()),axis=1)
        scipy.misc.imsave(configPara.samples_save_dir+'/train_image%d.png'%epoch,result)

    for epoch in range(1, 101):
        epoch_time = time.time()
        sum_loss=0
        n_iter=0
        for idx in range(0, 1000, 1):
            step_time = time.time()
            seq=random.randint(0, dataNum-1)
            imgs_input=train_imgs[seq]
            imgs_input2,imgs_target2=sampleImg2(imgs_input,nn2)
            loss, _ = sess.run([loss_forward2,optim_forward2],{input2: imgs_input2, target2: imgs_target2})

            sum_loss=sum_loss+loss
            n_iter += 1
        print("[*] Epoch [%2d/%2d] time: %4.4fs, loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time,sum_loss/n_iter))

        if (epoch % configPara.save_model_freq == 0):
            tf.train.Saver().save(sess, configPara.checkpoint_dir+'/model%d'% epoch)

            seq=random.randint(0, dataNum-1)
            imgs_input=train_imgs[seq]
            imgs_input2,imgs_target2=sampleImg2(imgs_input,configPara.nn1)
            out = sess.run(forward2, {input2:imgs_input2})
            result=np.concatenate((imgs_input2.squeeze(), out.squeeze()),axis=1)

            for i in range(9):
                seq=random.randint(0, dataNum-1)
                imgs_input=train_imgs[seq]
                imgs_input2,imgs_target2=sampleImg2(imgs_input,configPara.nn1)
                out = sess.run(forward2, {input2:imgs_input2})
                out=np.concatenate((imgs_input2.squeeze(), out.squeeze()),axis=1)
                result=np.concatenate((result, out),axis=0)

            scipy.misc.imsave(configPara.samples_save_dir+'/train_forward1%d.png'%epoch,result)

    for epoch in range(1, 10001):
        epoch_time = time.time()
        sum_loss=0
        n_iter=0
        for idx in range(0, 1000, 1):
            step_time = time.time()
            seq=random.randint(0, dataNum-1)
            imgs_input=train_imgs[seq]
            imgs_input1,imgs_target1=sampleImg(imgs_input,configPara.nn1)
            loss, _ = sess.run([loss_forward1,optim_forward1],{input1: imgs_input1, target1: imgs_target1})

            sum_loss=sum_loss+loss
            n_iter += 1
        print("[*] Epoch [%2d/%2d] time: %4.4fs, L1_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time,sum_loss/n_iter))

        if (epoch % configPara.save_model_freq == 0):
            tf.train.Saver().save(sess, configPara.checkpoint_dir+'/model%d'% epoch)

            seq=random.randint(0, dataNum-1)
            imgs_input=train_imgs[seq]
            imgs_input,imgs_target=sampleImg(imgs_input,configPara.nn1)
            out = sess.run(forward1, {input1:imgs_input})
            result=np.concatenate((imgs_input.squeeze(), out.squeeze()),axis=1)
            scipy.misc.imsave(configPara.samples_save_dir+'/train_forward2%d.png'%epoch,result)

        if epoch % configPara.test_freq==0:
            print("---------------------------test--------------------------------")
            runTest(test_imgs,sess)

def runTest(test_imgs,sess):
    for i in range(0, len(test_imgs), 1):
        inputImage=test_imgs[i]
        w,h=inputImage.shape

        n=8
        h2=h-h%n
        w2=w-w%n

        input_image_forward_large = tf.placeholder('float32', [1, w2, h2, 1], name='input_test')
        image_forward_large = generator(input_image_forward_large, reuse=True,net_name="net_forward")

        inputImage2=inputImage[0:w2,0:h2]
        #test_img = test_img[0:w2,0:h2]
        inputImage2=np.expand_dims(np.expand_dims(inputImage2,axis=0),axis=3)
        out = sess.run(image_forward_large, {input_image_forward_large:inputImage2})

        #np.save(configPara.test_save_dir+'/my_%d.txt' % i,out.squeeze())

        np.savetxt(configPara.test_save_dir+'/my_%d.txt' % i, out.squeeze(), delimiter=' ')
        np.savetxt(configPara.test_save_dir+'/input_%d.txt' % i, inputImage2.squeeze(), delimiter=' ')
        #np.savetxt(configPara.test_save_dir+'/target_%d.txt' % i, test_img, delimiter=' ')
        #out = scipy.misc.imresize(reScale(out.squeeze()), size=[w, h], interp='bicubic', mode=None)
        #out=np.concatenate((inputImage, out),axis=1)
        scipy.misc.imsave(configPara.test_save_dir+'/my_%d.png' % i,reScale(out.squeeze()))

def test():
    test_imgs = read_all_imgs(configPara.test_image_path, regx='.*.txt')
    #test_imgs = generate_noiseimgs(test_img,20)

    input_image_forward_large = tf.placeholder('float32', [1, 512, 512, 1], name='input_test')
    image_forward_large = generator(input_image_forward_large, reuse=False,net_name="net_forward")
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(configPara.checkpoint_dir)
    tf.train.Saver().restore(sess,checkpoint)

    runTest(test_imgs,sess)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise Exception("Unknow --mode")
