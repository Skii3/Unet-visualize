import tensorflow as tf
import PIL
import scipy.misc
import numpy as np
import re
import os
import random
from config import configPara
from skimage.util import random_noise
from scipy import ndimage

def read_all_imgs(path,regx='*.txt'):
    file_list = os.listdir(path)
    img_list = []
    for idx, f in enumerate(file_list):
       # if re.search(regx, f):
        img_list.append(f)
    if configPara.type != 0:
        img_list=sorted(img_list,key= lambda x:int(x[:-4]))
    else:
        img_list = sorted(img_list)
    imgs = []
    #for idx in range(0, 5):
    for idx in range(0, len(img_list)):
        img= np.loadtxt(path + img_list[idx])
        imgs.append(img)
    return imgs

def expand_imgs(imgs):
    for k in range(0,len(imgs)):
        img=imgs[k]
        h,w=img.shape
        for step in range(2,configPara.scale+1):
            #temp=img[0:h:step,0:w]
            #imgs.append(temp)
            temp=img[0:h,0:w:step]
            imgs.append(temp)
    for k in range(0,len(imgs)):
        scipy.misc.imsave(configPara.buffer_dir+'/train_%d.png' % k,imgs[k])
    return imgs

def addNoise(inputImage):
    mean_value = np.mean(np.abs(inputImage))
    noiseImage = inputImage + np.random.normal(0, random.randint(1, 40) * 1e-1 * mean_value, inputImage.shape)
    #sigma=random.randint(1, 10)
    #noise = np.random.normal(0, sigma*1e-2, img.shape)
    return noiseImage

def generate_noiseimgs(imgs,k):
    test_imgs=[]
    num=0
    for j in range(0,len(imgs)):
        for i in range(k):
            tmp = addNoise(imgs[j])
            test_imgs.append(tmp)
            np.savetxt(configPara.test_image_path+'/input_%d.txt' % num, tmp, delimiter=' ')
            #np.savetxt(configPara.test_image_path+'/target_%d.txt' % num, imgs[j], delimiter=' ')
            num=num+1
    for k in range(0,len(test_imgs)):
        scipy.misc.imsave(configPara.test_image_path+'/test_%d.png' % k,test_imgs[k])
    return test_imgs

def formatImg(img):
    img=myScale(np.float32(img))
    img=np.expand_dims(img,axis=0)
    return img

def sampleImg_clean(inputImg,nn,batch_num = 1):
    inputImage_all =[]
    targetImage_all = []
    w,h=inputImg.shape

    h_size=nn
    w_size=nn
    if h-h_size <= 0 or w-w_size <= 0:
        return None,None
    for i in range(batch_num):
        h_loc=random.randint(0, h-h_size)
        w_loc=random.randint(0, w-w_size)
        inputImage=inputImg[w_loc:w_loc+w_size,h_loc:h_loc+h_size]

        aug_type=random.randint(0, 12)
        if aug_type==1:
            inputImage=np.fliplr(inputImage);
        if aug_type==2:
            inputImage=np.flipud(inputImage);
        if aug_type==3:
            inputImage=np.rot90(inputImage,1);
        if aug_type==4:
            inputImage=np.rot90(inputImage,2);
        if aug_type==5:
            inputImage=np.rot90(inputImage,3);
        if aug_type==6:
            inputImage=np.flipud(np.rot90(inputImage,1));
        if aug_type==7:
            inputImage=np.flipud(np.rot90(inputImage,2));
        if aug_type==8:
            inputImage=np.flipud(np.rot90(inputImage,3));
        if aug_type==9:
            inputImage=np.fliplr(np.rot90(inputImage,1));
        if aug_type==10:
            inputImage=np.fliplr(np.rot90(inputImage,2));
        if aug_type==11:
            inputImage=np.fliplr(np.rot90(inputImage,3));

        #sigma = random.randint(1, 10)

        intensity_aug=random.randint(1, 5)
        if intensity_aug==2:
            inputImage = inputImage * np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
            inputImage = inputImage / np.max(inputImage)
        if intensity_aug==3:
            inputImage = inputImage / np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
            inputImage = inputImage / np.max(inputImage)

        mean_value = np.mean(np.abs(inputImage))
        noiseImage = inputImage + np.random.normal(0, random.randint(1, 25) * 1e-1 * mean_value, inputImage.shape)

        noiseImage=np.expand_dims(np.expand_dims(noiseImage,axis=0),axis=3)
        inputImage=np.expand_dims(np.expand_dims(inputImage,axis=0),axis=3)

        #noiseImage = (noiseImage - np.mean(noiseImage)) / np.std(noiseImage)
        #inputImage = (inputImage - np.mean(noiseImage)) / np.std(noiseImage)
        inputImage_all.append(noiseImage)
        targetImage_all.append(inputImage)
    return targetImage_all,targetImage_all

def sampleImg(inputImg,nn):
    w,h=inputImg.shape

    h_size=nn
    w_size=nn
    if h-h_size <= 0 or w-w_size <= 0:
        return None,None
    h_loc=random.randint(0, h-h_size)
    w_loc=random.randint(0, w-w_size)
    inputImage=inputImg[w_loc:w_loc+w_size,h_loc:h_loc+h_size]

    aug_type=random.randint(0, 12)
    if aug_type==1:
        inputImage=np.fliplr(inputImage);
    if aug_type==2:
        inputImage=np.flipud(inputImage);
    if aug_type==3:
        inputImage=np.rot90(inputImage,1);
    if aug_type==4:
        inputImage=np.rot90(inputImage,2);
    if aug_type==5:
        inputImage=np.rot90(inputImage,3);
    if aug_type==6:
        inputImage=np.flipud(np.rot90(inputImage,1));
    if aug_type==7:
        inputImage=np.flipud(np.rot90(inputImage,2));
    if aug_type==8:
        inputImage=np.flipud(np.rot90(inputImage,3));
    if aug_type==9:
        inputImage=np.fliplr(np.rot90(inputImage,1));
    if aug_type==10:
        inputImage=np.fliplr(np.rot90(inputImage,2));
    if aug_type==11:
        inputImage=np.fliplr(np.rot90(inputImage,3));

    #sigma = random.randint(1, 10)

    intensity_aug=random.randint(1, 5)
    if intensity_aug==2:
        inputImage = inputImage * np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
        inputImage = inputImage / np.max(inputImage)
    if intensity_aug==3:
        inputImage = inputImage / np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
        inputImage = inputImage / np.max(inputImage)

    mean_value = np.mean(np.abs(inputImage))
    noiseImage = inputImage + np.random.normal(0, random.randint(1, 25) * 1e-1 * mean_value, inputImage.shape)

    noiseImage=np.expand_dims(np.expand_dims(noiseImage,axis=0),axis=3)
    inputImage=np.expand_dims(np.expand_dims(inputImage,axis=0),axis=3)

    #noiseImage = (noiseImage - np.mean(noiseImage)) / np.std(noiseImage)
    #inputImage = (inputImage - np.mean(noiseImage)) / np.std(noiseImage)
    return noiseImage,inputImage

def sampleImg2(inputImg,nn):
    w,h=inputImg.shape

    h_size=nn
    w_size=nn
    h_loc=random.randint(0, h-h_size)
    w_loc=random.randint(0, w-w_size)
    inputImage=inputImg[w_loc:w_loc+w_size,h_loc:h_loc+h_size]
    if configPara.if_aug == True:
        aug_type=random.randint(0, 12)
        if aug_type==1:
            inputImage=np.fliplr(inputImage);
        if aug_type==2:
            inputImage=np.flipud(inputImage);
        if aug_type==3:
            inputImage=np.rot90(inputImage,1);
        if aug_type==4:
            inputImage=np.rot90(inputImage,2);
        if aug_type==5:
            inputImage=np.rot90(inputImage,3);
        if aug_type==6:
            inputImage=np.flipud(np.rot90(inputImage,1));
        if aug_type==7:
            inputImage=np.flipud(np.rot90(inputImage,2));
        if aug_type==8:
            inputImage=np.flipud(np.rot90(inputImage,3));
        if aug_type==9:
            inputImage=np.fliplr(np.rot90(inputImage,1));
        if aug_type==10:
            inputImage=np.fliplr(np.rot90(inputImage,2));
        if aug_type==11:
            inputImage=np.fliplr(np.rot90(inputImage,3));

        #sigma = random.randint(1, 10)

        intensity_aug=random.randint(1, 5)
        if intensity_aug==2:
            inputImage = inputImage * np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
            inputImage = inputImage / np.max(inputImage)
        if intensity_aug==3:
            inputImage = inputImage / np.sqrt(np.sqrt(np.abs(inputImage)+1e-12))
            inputImage = inputImage / np.max(inputImage)


    targetImage = np.zeros(inputImage.shape)
    #if image_type==1:
    mean_value = np.mean(np.abs(inputImage))
    noiseImage = inputImage + np.random.normal(0, random.randint(1, 25) * 1e-1 * mean_value, inputImage.shape)
    temp = inputImage - ndimage.uniform_filter(inputImage, size=3)
    targetImage[np.where(np.abs(temp)>0.02)]=1
    #stre=ndimage.generate_binary_structure(1,1)
    #targetImage = ndimage.binary_dilation(targetImage)
    #else:
    #    noiseImage = np.random.normal(0, random.randint(1, 10) * 1e-2, inputImage.shape)
    #noiseImage = np.abs(noiseImage)
    noiseImage=np.expand_dims(np.expand_dims(noiseImage,axis=0),axis=3)
    targetImage=np.expand_dims(np.expand_dims(targetImage,axis=0),axis=3)

    return noiseImage,targetImage

'''
def sampleImg(inputImg,outImg,nn):
    outputImage=PIL.Image.fromarray(np.uint8(outImg))

    w,h=outputImage.size
    h_size=nn
    w_size=nn

    if configPara.if_scale:

        scale=configPara.scale
        max_h=nn*scale
        max_w=nn*scale
        min_h=nn/scale
        min_w=nn/scale
        if min_h>h:
          min_h=h
        if min_w>w:
          min_w=w
        if max_h>h:
          max_h=h
        if max_w>w:
          max_w=w
        h_size=random.randint(min_h, max_h+1)
        w_size=random.randint(min_w, max_w+1)
        #h_size=random.randint(nn, h+1)
        #w_size=random.randint(nn, w+1)
    h_loc=random.randint(0, h-h_size+1)
    w_loc=random.randint(0, w-w_size+1)
    box=(w_loc,h_loc,w_loc+w_size,h_loc+h_size)
    outputImage=outputImage.crop(box)

    if configPara.if_scale:
        imgSize=[nn,nn]
        if not h_size==nn or not w_size==nn:
            outputImage=outputImage.resize(imgSize,PIL.Image.BICUBIC)

    aug_type=random.randint(0, 7)
    if aug_type==1:
        outputImage=outputImage.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    if aug_type==2:
        outputImage=outputImage.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    if aug_type==3:
        outputImage=outputImage.transpose(PIL.Image.ROTATE_90)
    if aug_type==4:
        outputImage=outputImage.transpose(PIL.Image.ROTATE_180)
    if aug_type==5:
        outputImage=outputImage.transpose(PIL.Image.ROTATE_270)


    inputImage = PIL.Image.fromarray(np.uint8(random_noise(np.asarray(outputImage), seed=42, var=random.randint(0, 20)*1e-4)*255.0))

    inputImage=np.asarray(inputImage)
    outputImage=np.asarray(outputImage)
    #inputImage = random_noise(outputImage, seed=42, var=random.randint(1,10)*1e-4)

    inputImage=myScale(np.float32(inputImage))
    outputImage=myScale(np.float32(outputImage))

    inputImage=np.expand_dims(inputImage,axis=0)
    outputImage=np.expand_dims(outputImage,axis=0)

    return inputImage,outputImage
'''
def downsample(x):
    [w,h,c]=x.shape
    rate=configPara.rate
    x =  scipy.misc.imresize(x, size=[w/rate/2, h/rate/2], interp='bicubic', mode=None)
    x =  scipy.misc.imresize(x, size=[w, h], interp='bicubic', mode=None)
    return x

def myScale(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def reScale(x):
    x = x+1
    x = x*(255. / 2.)
    return x

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def imageDiff(img1,img2):
    pos = tf.constant(np.identity(3), dtype=tf.float32)
    neg = -1 * pos
    filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
    filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
    strides = [1, 1, 1, 1]  # stride of (1, 1)
    padding = 'SAME'

    gen_dx = tf.abs(tf.nn.conv2d(img1, filter_x, strides, padding=padding))
    gen_dy = tf.abs(tf.nn.conv2d(img1, filter_y, strides, padding=padding))

    gt_dx = tf.abs(tf.nn.conv2d(img2, filter_x, strides, padding=padding))
    gt_dy = tf.abs(tf.nn.conv2d(img2, filter_y, strides, padding=padding))

    grad_diff_x = tf.abs(gt_dx - gen_dx)
    grad_diff_y = tf.abs(gt_dy - gen_dy)
    return grad_diff_x,grad_diff_y
