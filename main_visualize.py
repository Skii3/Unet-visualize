#! /usr/bin/python

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

graph2 = tf.Graph()
visual_target = 'c'
with graph2.as_default():
    # construct network again
    ngf = 64
    ind = 0
    input2 = tf.placeholder('float32', [1, nn1, nn1, 1], name='input2_image_forward')
    if visual_target == 'd2':
        forward2 = generator2_2(input2, reuse=False, net_name="net2_forward")
    elif visual_target == 'd3':
        forward2 = generator(input2, reuse=False, net_name="net2_forward")
    else:
        forward2 = generator2(input2, reuse=False, net_name="net2_forward")
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph2)
    sess2.run(tf.global_variables_initializer())
    train_vars2 = tf.trainable_variables()
    vars_forward2_name = [var.name for var in train_vars2 if 'net2_forward' in var.name]
    for i in range(0, len(vars_forward2_name), 1):

        if (("d2" in vars_forward2_name[i] or "d3" in vars_forward2_name[i]) and visual_target != 'd3') \
                or ("d3" in vars_forward2_name[i] and visual_target == 'd2'):
            weight_temp = weight_list[i]
            weight_temp2 = weight_temp[:, :, :, 0:np.int(np.shape(weight_temp)[-1]/2)]
            temp = graph2.get_tensor_by_name(vars_forward2_name[i])
            temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[i]), weight_temp2)
            sess2.run(temp)
        else:
            temp = graph2.get_tensor_by_name(vars_forward2_name[i])
            temp = tf.assign(graph2.get_tensor_by_name(vars_forward2_name[i]), weight_list[i])
            sess2.run(temp)

graph3 = tf.Graph()
with graph3.as_default():
    # construct network again
    ngf = 64
    ind = 0
    input3 = tf.placeholder('float32', [1, nn1, nn1, 1], name='input3_image_forward')
    forward3 = generator3(input3, reuse=False, net_name="net3_forward")
    sess3 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False), graph=graph3)
    sess3.run(tf.global_variables_initializer())
    train_vars3 = tf.trainable_variables()
    vars_forward3_name = [var.name for var in train_vars3 if 'net3_forward' in var.name]
    # c1/filter
    temp = graph3.get_tensor_by_name(vars_forward3_name[0])
    temp = tf.assign(graph3.get_tensor_by_name(vars_forward3_name[0]), weight_list[0])
    sess3.run(temp)
    # d3/filter
    temp = graph3.get_tensor_by_name(vars_forward3_name[1])
    weight_temp = weight_list[13]
    weight_temp2 = weight_temp[:, :, :, np.int(np.shape(weight_temp)[-1]/2): np.shape(weight_temp)[-1]]
    temp = tf.assign(graph3.get_tensor_by_name(vars_forward3_name[1]), weight_temp2)
    sess3.run(temp)
    # c1/filter
    temp = graph3.get_tensor_by_name(vars_forward3_name[2])
    temp = tf.assign(graph3.get_tensor_by_name(vars_forward3_name[2]), weight_list[14])
    sess3.run(temp)

# sample result of this model
train_imgs = read_all_imgs(configPara.train_image_path, regx='.*.txt')
train_imgs = expand_imgs(train_imgs)

dataNum = len(train_imgs)
seq = random.randint(0, dataNum - 1)
imgs_input = train_imgs[0]
mean_input = np.mean(imgs_input)
print(mean_input)
imgs_input1, imgs_target1 = sampleImg(imgs_input, configPara.nn1)


'''     reconstructed model     '''
net_layers = [op.name for op in graph2.get_operations() if (op.type == 'Conv2D'
                                                           or ( "conv2d_transpose" in op.name and "output_shape"
                                                                not in op.name))
              and "net2_forward" in op.name or "input2_image_forward" in op.name]
net_feature_map = []
for i in range(0, len(net_layers), 1):
    tensor = graph2.get_tensor_by_name(net_layers[i]+':0')
    temp = sess2.run(tensor, feed_dict={input2:imgs_input1})
    str = net_layers[i]
    str = str.replace("/", "_")
    file_name = configPara.samples_save_dir + '/' + str
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    for j in range(0, temp.shape[-1], 1):
        temp2 = temp[:, :, :, j]
        scipy.misc.imsave(file_name + '/channel_%d.png'%j, temp2.squeeze())

out2 = sess2.run(forward2, {input2: imgs_input1})
scipy.misc.imsave(configPara.samples_save_dir + '/2denoise.bmp', out2.squeeze())
scipy.misc.imsave(configPara.samples_save_dir + '/2noisedata.bmp', imgs_input1.squeeze())
scipy.misc.imsave(configPara.samples_save_dir + '/2clean.bmp', imgs_target1.squeeze())

'''     reconstructed2 model     '''
net_layers = [op.name for op in graph3.get_operations() if (op.type == 'Conv2D'
                                                           or ( "conv2d_transpose" in op.name and "output_shape"
                                                                not in op.name))
              and "net3_forward" in op.name or "input3_image_forward" in op.name]
net_feature_map = []
for i in range(0, len(net_layers), 1):
    tensor = graph3.get_tensor_by_name(net_layers[i]+':0')
    temp = sess3.run(tensor, feed_dict={input3:imgs_input1})
    str = net_layers[i]
    str = str.replace("/", "_")
    file_name = configPara.samples_save_dir + '/' + str
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    for j in range(0, temp.shape[-1], 1):
        temp2 = temp[:, :, :, j]
        scipy.misc.imsave(file_name + '/channel_%d.png'%j, temp2.squeeze())

out3 = sess3.run(forward3, {input3: imgs_input1})
scipy.misc.imsave(configPara.samples_save_dir + '/3denoise.bmp', out3.squeeze())
scipy.misc.imsave(configPara.samples_save_dir + '/3noisedata.bmp', imgs_input1.squeeze())
scipy.misc.imsave(configPara.samples_save_dir + '/3clean.bmp', imgs_target1.squeeze())


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

type_cross_connect = 0
graph3 = tf.Graph()
if type_cross_connect == 0:
    graph3 = graph
elif type_cross_connect == 1:
    graph3 = graph2


# visualize first layer
img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    with graph3.as_default():
        t_score = tf.reduce_mean(t_obj)  # defining the optimization objective
        if type_cross_connect == 1:
            t_grad = tf.gradients(t_score, input2)[0]  # behold the power of automatic differentiation!
        else:
            t_grad = tf.gradients(t_score, input1)[0]

        img = img0.copy()
        for i in range(iter_n):
            if type_cross_connect == 1:
                g, score = sess2.run([t_grad, t_score], {input2: img})
            else:
                g, score = sess.run([t_grad, t_score], {input1: img})
            # normalizing the gradient, so the same step size should work
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step
            if i % 100 == 0:
                print(i, ' ', score, ' \n')
            if iter_n <= 100 and i % 10 == 0:
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

        gTemp = lap_normalize(g_input, scale_n=4)
        #gTemp = tf.nn.conv2d(g_input, k5x5, [1, 2, 2, 1], 'SAME')
        #gTemp = tf.nn.conv2d_transpose(gTemp, k5x5 * 4, tf.shape(g_input), [1, 2, 2, 1])
        img = img0.copy()
        for i in range(iter_n):
            if type_cross_connect == 0:
                g, score = sess.run([t_grad, t_score], {input1: img})
                g_temp = g / np.maximum(np.std(g),1e-10)
                g = sess.run(gTemp, feed_dict={g_input:g})
                g_temp2 = g / np.maximum(np.std(g), 1e-10)
                gg = np.concatenate((g_temp.squeeze(),g_temp2.squeeze()),axis=1)
                scipy.misc.imsave('./g_glap.bmp', gg.squeeze())
            else:
                g, score = sess2.run([t_grad, t_score], {input2: img})
                g = sess2.run(gTemp, feed_dict={g_input: g})
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
            if type_cross_connect == 0:
                g, score, sum, tv = sess.run([t_grad, t_score, reduce_sum, tvDiff_loss_forward], {input1: img})
            else:
                g, score, sum, tv = sess2.run([t_grad, t_score, reduce_sum, tvDiff_loss_forward], {input2: img})
            #g = sess.run(gTemp, feed_dict={g_input:g})
            g /= g.std() + 1e-8  # for different layers and networks
            img += g * step * 0.001
            if i % 100 == 0:
                print(i, ' ', score, ' ', sum, ' ', tv*lam, ' \n')
    return img

visual_type = 1
if 'd' in visual_target:
    visual_type = 2

if type_cross_connect == 1:
    layers = [op.name for op in graph3.get_operations() if op.type == 'Conv2D' and "net2_forward" in op.name]
    if visual_type == 2:
        layers = [op.name for op in graph3.get_operations() if "conv2d_transpose" in op.name and "output_shape" not in op.name]
else:
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
    layer = layers[3]
    img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input

    for channel in range(0, 1, 1):
        for index1 in range(40, 128, 32):
            for index2 in range(40, 128, 32):
                iter = 2000
                img = render_tv(T(layer)[:, index1, index2, channel], img_noise, iter,1,1000)
                str = layer
                str = str.replace("/", "_")
                scipy.misc.imsave('naive_feature_output_tv/%s_%d_%d_%d_iter%d.png' % (str, channel, index1, index2, iter), img.squeeze())
elif visual_type == 1:      # c1,c2,c3
    # visualize every channel with sliced shape in each layer

    for ind in range(2,3,1):
        img_all = []
        layer = layers[ind]
        feature_num = feature_nums[ind]
        img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input
        #img_noise = np.zeros([1, nn1, nn1, 1]) + mean_input
        for channel in range(0, feature_num, 1):
            index1 = 8 * (3-ind)
            index2 = 8 * (3-ind)
            iter = 2000
            if ind == 3:
                index1 = 30
                index2 = 30
                iter = 1000

            if ind <= 3:
                #img = render_naive(T(layer)[:, index1, index2, channel], img_noise, iter)
                img = render_lapnorm(T(layer)[:, index1, index2, channel], img_noise, iter,step=1.0, lap_n=4)
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
    for ind in range(2,3,1):
        layer = layers[ind]
        feature_num = feature_nums[ind]
        img_noise = np.random.uniform(size=(1, nn1, nn1, 1)) + mean_input
        #img_noise = np.zeros([1, nn1, nn1, 1]) + mean_input
        img_all = []
        iter = 100
        for channel in range(0, feature_num, 1):
            index1 = (ind + 1) * 16
            index2 = (ind + 1) * 16

            if ind == 1:
                index1 = 10
                index2 = 10
            elif ind == 2:
                index1 = 65
                index2 = 65


            img = render_naive(T(layer)[:, index1, index2, channel], img_noise, iter)
            #img = render_tv(T(layer)[:, index1, index2, channel], img_noise, iter, 1.0, 10)
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
                file_name+'/%s_%d_%d_%d_iter%d.png' % (str, channel, index1, index2, iter),
                img_save.squeeze())
            img_all.append(img_save.squeeze())
        sio.savemat('naive_feature_all_d%d_iter%d.mat' % (ind + 1,iter), {'img_all': img_all})
