#! /usr/bin/python
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from model import *
from utils import *
import scipy.io as sio
from config import configPara
import time
import scipy
import functools
import os
# construct net
graph = tf.get_default_graph()
batch_size = 50
nn1 = configPara.nn1
input1 = tf.placeholder('float32', [batch_size, nn1, nn1, 1], name='input_image_forward1')
target1 = tf.placeholder('float32', [batch_size, nn1, nn1, 1], name='target_image_forward1')
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
train_imgs_final = []
for i in range(dataNum):
    temp = train_imgs[i]
    if np.shape(temp)[0] > nn1 and np.shape(temp)[1] > nn1:
        train_imgs_final.append(temp)
dataNum = len(train_imgs_final)
net_layers = [op.name for op in graph.get_operations() if (op.type == 'Conv2D'
                                                           or ( "conv2d_transpose" in op.name and "output_shape"
                                                                not in op.name))
              and "net_forward" in op.name or "input_image_forward1" in op.name ]
LEN_LAYER = len(net_layers)
for i in range(1, 6, 1):
    tensor = graph.get_tensor_by_name(net_layers[i]+':0')
    len_channel = tensor.shape[-1]
    str = net_layers[i].replace("/", "_")
    if not os.path.exists(str):
        os.mkdir(str)

    num_keep = 6

    if 'c1' in net_layers[i]:
        index1 = 1
        img_size = 2 * index1 - 1 + 4 - (2 * index1 - 1)
        #img_save = img[:, 2 * index1 - 1:2 * index1 - 1 + 4, 2 * index2 - 1:2 * index2 - 1 + 4, :]
    elif 'c2' in net_layers[i]:
        index1 = 1
        img_size = 2*(2*index1-1+3)-1+4 - (2*(2*index1-1)-1)
    elif 'c3' in net_layers[i]:
        index1 = 1
        img_size = 2 * (2 * (2 * index1 - 1 + 3) - 1 + 3) - 1 + 4 - (2 * (2 * (2 * index1 - 1) - 1) - 1)
    elif 'd1' in net_layers[i]:
        index1 = 1
        a = np.int(np.ceil((index1 + 1) / 2))
        img_size = 8*a+14+1 - (8*a-15)
    elif 'd2' in net_layers[i]:
        index1 = 1
        b1 = np.int(np.ceil((index1 + 1) / 2) - 1)
        a1 = np.int(np.ceil((b1 + 1) / 2))
        b2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
        a2 = np.int(np.ceil((b2 + 1) / 2))
        img_size = 8 * a2 + 14 + 1 - ( 8 * a1 - 15)
    elif 'd3' in net_layers[i]:
        index1 = 1
        c1 = np.int(np.ceil((index1 + 1) / 2) - 1)
        c2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
        b1 = np.int(np.ceil((c1 + 1) / 2) - 1)
        b2 = np.int(np.ceil((c2 + 1) / 2) - 1 + 1)
        a1 = np.int(np.ceil((b1 + 1) / 2))
        a2 = np.int(np.ceil((b2 + 1) / 2))
        img_size = 8 * a2 + 14 + 1 - (8 * a1 - 15)
    keep_list_all = np.zeros([len_channel,num_keep])
    img_list_all = np.zeros([len_channel,num_keep,img_size,img_size])
    for idx in range(0, 10000, 1):
        print idx
        step_time = time.time()
        seq = random.randint(0, dataNum - 1)
        imgs_input = train_imgs_final[seq]
        imgs_input1, imgs_target1 = sampleImg_clean(imgs_input, configPara.nn1,batch_size)
        # scipy.misc.imsave(configPara.buffer_dir+'/trainImg%d%d.png'%(epoch,idx),imgs_input1.squeeze())
        imgs_input1 = np.reshape(imgs_input1, newshape=[batch_size,nn1,nn1,1])
        imgs_target1 = np.reshape(imgs_target1, newshape=[batch_size, nn1, nn1, 1])
        feature_map \
            = sess.run([tensor], \
                       {input1: imgs_input1, target1: imgs_target1})
        feature_map = np.array(feature_map).squeeze()
        max_value_all = np.max(feature_map,axis=(0,1,2))
        for j in range(len_channel):
            max_value = max_value_all[j]
            for k in range(num_keep):
                if max_value > keep_list_all[j,k]:
                    index = np.where(feature_map[:,:,:,j] == max_value)
                    index0 = index[0][0]
                    index1 = index[1][0]
                    index2 = index[2][0]
                    h = tensor.shape[-3]
                    w = tensor.shape[-2]
                    [ind_img11, ind_list11, ind_img12, ind_list12] = [0, 0, img_size, img_size]
                    [ind_img21, ind_list21, ind_img22, ind_list22] = [0, 0, img_size, img_size]
                    if 'c1' in net_layers[i]:
                        ind_img12 = 2 * index1 - 1 + 4
                        ind_img22 = 2 * index2 - 1 + 4
                        ind_img11 = 2 * index1 - 1
                        ind_img21 = 2 * index2 - 1
                    elif 'c2' in net_layers[i]:
                        ind_img12 = 2 * (2 * index1 - 1 + 3) - 1 + 4
                        ind_img22 = 2 * (2 * index2 - 1 + 3) - 1 + 4
                        ind_img11 = 2 * (2 * index1 - 1) - 1
                        ind_img21 = 2 * (2 * index2 - 1) - 1
                    elif 'c3' in net_layers[i]:
                        ind_img12 = 2 * (2 * (2 * index1 - 1 + 3) - 1 + 3) - 1 + 4
                        ind_img22 = 2 * (2 * (2 * index2 - 1 + 3) - 1 + 3) - 1 + 4
                        ind_img11 = 2 * (2 * (2 * index1 - 1) - 1) - 1
                        ind_img21 = 2 * (2 * (2 * index2 - 1) - 1) - 1
                    elif 'd1' in net_layers[i]:
                        a = np.int(np.ceil((index1 + 1) / 2))
                        ind_img12 = 8 * a + 14 + 1
                        b = np.int(np.ceil((index2 + 1) / 2))
                        ind_img22 = 8 * b + 14 + 1
                        a = np.int(np.ceil((index1 + 1) / 2))
                        ind_img11 = 8 * a - 15
                        b = np.int(np.ceil((index2 + 1) / 2))
                        ind_img21 = 8 * b - 15
                    elif 'd2' in net_layers[i]:
                        b2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
                        a2 = np.int(np.ceil((b2 + 1) / 2))
                        ind_img12 = 8 * a2 + 14 + 1
                        b22 = np.int(np.ceil((index2 + 1) / 2) - 1 + 1)
                        a22 = np.int(np.ceil((b22 + 1) / 2))
                        ind_img22 = 8 * a22 + 14 + 1
                        b1 = np.int(np.ceil((index1 + 1) / 2) - 1)
                        a1 = np.int(np.ceil((b1 + 1) / 2))
                        ind_img11 = 8 * a1 - 15
                        b11 = np.int(np.ceil((index2 + 1) / 2) - 1)
                        a11 = np.int(np.ceil((b11 + 1) / 2))
                        ind_img21 = 8 * a11 - 15
                    elif 'd3' in net_layers[i]:
                        c2 = np.int(np.ceil((index1 + 1) / 2) - 1 + 1)
                        b2 = np.int(np.ceil((c2 + 1) / 2) - 1 + 1)
                        a2 = np.int(np.ceil((b2 + 1) / 2))
                        ind_img12 = 8 * a2 + 14 + 1
                        c22 = np.int(np.ceil((index2 + 1) / 2) - 1 + 1)
                        b22 = np.int(np.ceil((c22 + 1) / 2) - 1 + 1)
                        a22 = np.int(np.ceil((b22 + 1) / 2))
                        ind_img22 = 8 * a22 + 14 + 1
                        c1 = np.int(np.ceil((index1 + 1) / 2) - 1)
                        b1 = np.int(np.ceil((c1 + 1) / 2) - 1)
                        a1 = np.int(np.ceil((b1 + 1) / 2))
                        ind_img11 = 8 * a1 - 15
                        c11 = np.int(np.ceil((index2 + 1) / 2) - 1)
                        b11 = np.int(np.ceil((c11 + 1) / 2) - 1)
                        a11 = np.int(np.ceil((b11 + 1) / 2))
                        ind_img21 = 8 * a11 - 15
                    if ind_img11 < 0 or ind_img21 < 0 or ind_img12 > nn1 or ind_img22 > nn1:
                        continue
                    if ind_img11 < 0:
                        ind_img11 = 0
                        ind_list11 = img_size - (ind_img12 - ind_img11)
                        ind_list12 = img_size
                    if ind_img21 < 0:
                        ind_img21 = 0
                        ind_list21 = img_size - (ind_img22 - ind_img21)
                        ind_list22 = img_size
                    if ind_img12 > nn1:
                        ind_img12 = nn1
                        ind_list11 = 0
                        ind_list12 = (ind_img12 - ind_img11)
                    if ind_img22 > nn1:
                        ind_img22 = nn1
                        ind_list21 = 0
                        ind_list22 = (ind_img22 - ind_img21)
                    ind_list12 = ind_list11 + (ind_img12-ind_img11)
                    ind_list22 = ind_list21 + (ind_img22 - ind_img21)
                    temp = imgs_input1[index0, ind_img11:ind_img12, ind_img21:ind_img22]
                    img_list_all[j, k, :, :] = np.zeros([img_size,img_size])
                    img_list_all[j,k, ind_list11:ind_list12, ind_list21:ind_list22] = temp.squeeze()
                    temp_mean = np.mean(temp)
                    keep_list_all[j, k + 1:] = keep_list_all[j, k:num_keep - 1]
                    keep_list_all[j, k] = max_value
                    break
    for j in range(len_channel):
        print j
        print(keep_list_all[j,:])

        file_name = str + '/channel_%d' % j
        if not os.path.exists(file_name):
                os.mkdir(file_name)
        w_sum = np.sum(keep_list_all[j,:])
        img_all = np.zeros([num_keep/2*img_size,2*img_size])
        for k in range(num_keep):
            scipy.misc.imsave(file_name + '/%d.bmp' % (k),img_list_all[j,k,:,:].squeeze())
            img_list_all[j,k,:,:] = img_list_all[j,k,:,:] * keep_list_all[j,k] / w_sum
        k = 0
        for ii in range(0,num_keep/2,1):
            for jj in range(0,2,1):
                img_all[ii*img_size:(ii+1)*img_size,jj*img_size:(jj+1)*img_size] = img_list_all[j,k,:,:].squeeze()
                k =k + 1
        scipy.misc.imsave(file_name + '/sum.bmp', img_all.squeeze())
        img_mean = np.mean(img_list_all[j,:,:,:],axis=0)
        scipy.misc.imsave(file_name + '/%d_sum.bmp' % (k+1), img_mean.squeeze())
        np.savetxt(file_name + '/value.txt', keep_list_all[j,:], delimiter=' ')





