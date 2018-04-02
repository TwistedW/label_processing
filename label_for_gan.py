#-*- coding: utf-8 -*-
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
import sys
import pickle
from time import gmtime, strftime
from glob import glob
import matplotlib.pyplot as plt
import os, gzip

import tensorflow as tf
import tensorflow.contrib.slim as slim

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape(60000)

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape(10000)

    trY = np.asarray(trY) #将训练标签用数据保留
    teY = np.asarray(teY) #将测试标签用数据保留

    X = np.concatenate((trX, teX), axis=0) #将训练数据和测试数据拼接 (70000, 28, 28, 1)
    y = np.concatenate((trY, teY), axis=0).astype(np.int) #将训练标签和测试标签拼接 (70000)

    seed = 547
    np.random.seed(seed) #确保每次生成的随机数相同
    np.random.shuffle(X) #将mnist数据集中数据的位置打乱
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    #创建了(70000,10)的标签记录，并且根据mnist已有标签记录相应的10维数组
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    #返回归一化的数据和标签数组
    return X / 255., y_vec

def load_cifar10(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    X_train = []
    Y_train = []

    dirname = './data/cifar/cifar-10-batches-py'

    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        if i == 1:
            X_train = data
            Y_train = labels
        else:
            X_train = np.concatenate([X_train, data], axis=0)
            Y_train = np.concatenate([Y_train, labels], axis=0)

    fpath = os.path.join(dirname, 'test_batch')
    X_test, Y_test = load_batch(fpath)

    X_train = np.dstack((X_train[:, :1024], X_train[:, 1024:2048],
                         X_train[:, 2048:])) / 255.
    X_train = np.reshape(X_train, [-1, 32, 32, 3])
    X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048],
                        X_test[:, 2048:])) / 255.
    X_test = np.reshape(X_test, [-1, 32, 32, 3])

    X = np.concatenate((X_train, X_test), axis=0)

    # if one_hot:
    #    Y_train = to_categorical(Y_train, 10)
    #    Y_test = to_categorical(Y_test, 10)

    Y_test = np.asarray(Y_test)
    y = np.concatenate((Y_train, Y_test), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed) #确保每次生成的随机数相同
    np.random.shuffle(X) #将mnist数据集中数据的位置打乱
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    #创建了(70000,10)的标签记录，并且根据mnist已有标签记录相应的10维数组
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    #返回归一化的数据和标签数组
    return X, y_vec

def load_svhn(dataset_name):
    data_dir = os.path.join("./data/svhn", dataset_name)

    mat = sp.io.loadmat(data_dir)
    x_train = mat['X']

    x_train = np.transpose(x_train, axes=[3, 0, 1, 2])
    x_train = (x_train / 255.0).astype('float32')

    indices = mat['y']
    indices = np.squeeze(indices)
    indices[indices == 10] = 0
    y_train = np.zeros((len(indices), 10))
    y_train[np.arange(len(indices)), indices] = 1
    y_train = y_train.astype('float32')

    seed = 547
    np.random.seed(seed)  # 确保每次生成的随机数相同
    np.random.shuffle(x_train)  # 将mnist数据集中数据的位置打乱
    np.random.seed(seed)
    np.random.shuffle(y_train)
    return x_train, y_train

def load_Hair(n_classes, dataset_name, input_fname_pattern):
    #just for your own file
    L1 = glob(os.path.join("./data", dataset_name, "Black_Hair", input_fname_pattern))
    L2 = glob(os.path.join("./data", dataset_name, "Blond_Hair", input_fname_pattern))
    L3 = glob(os.path.join("./data", dataset_name, "Brown_Hair", input_fname_pattern))
    L4 = glob(os.path.join("./data", dataset_name, "Gray_Hair",  input_fname_pattern))
    data_X = np.concatenate((L1, L2), axis=0)
    data_X = np.concatenate((data_X, L3), axis=0)
    data_X = np.concatenate((data_X, L4), axis=0)
    ground_truths = []
    # Python中对于无需关注其实际含义的变量可以用_代替，这就和for  i in range(5)一样
    for _ in range(len(L1)):
        label_index = 0
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L2)):
        label_index = 1
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L3)):
        label_index = 2
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L4)):
        label_index = 3
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    seed = 547
    np.random.seed(seed)  # 确保每次生成的随机数相同
    np.random.shuffle(data_X)  # 将mnist数据集中数据的位置打乱
    np.random.seed(seed)
    np.random.shuffle(ground_truths)
    return data_X, ground_truths

def load_batch(fpath):
    with open(fpath, 'rb') as f:
        if sys.version_info > (3, 0):
            # Python3
            d = pickle.load(f, encoding='latin1')
        else:
            # Python2
            d = pickle.load(f)
    data = d["data"]
    labels = d["labels"]
    return data, labels

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

#显示所有变量的tensor类型
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    # 此处的np.argmax(id, 1)是用来判断此处的类别到底是几，如np.argmax([[0,0,1,0,0,0,0,0,0,0]],1)=2,输出最大的数所在的第二维度数字
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)