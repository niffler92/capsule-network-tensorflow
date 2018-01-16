import os
import scipy
import numpy as np
import tensorflow as tf


def load_data(dataset, batch_size, is_training=True):
    if dataset == 'mnist':
        path = os.path.join('data', 'mnist')
        if is_training:
            fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

            fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            trainY = loaded[8:].reshape((60000)).astype(np.int32)

            trX = trainX[:55000] / 255.
            trY = trainY[:55000]

            valX = trainX[55000:, ] / 255.
            valY = trainY[55000:]

            num_tr_batch = 55000 // batch_size
            num_val_batch = 5000 // batch_size

            return trX, trY, num_tr_batch, valX, valY, num_val_batch
        else:
            fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

            fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teY = loaded[8:].reshape((10000)).astype(np.int32)

            num_te_batch = 10000 // batch_size
            return teX / 255., teY, num_te_batch
    elif dataset == 'fashion_mnist':
        path = os.path.join('data', 'fashion-mnist')
        if is_training:
            fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

            fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            trainY = loaded[8:].reshape((60000)).astype(np.int32)

            trX = trainX[:55000] / 255.
            trY = trainY[:55000]

            valX = trainX[55000:, ] / 255.
            valY = trainY[55000:]

            num_tr_batch = 55000 // batch_size
            num_val_batch = 5000 // batch_size

            return trX, trY, num_tr_batch, valX, valY, num_val_batch
        else:
            fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

            fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            teY = loaded[8:].reshape((10000)).astype(np.int32)

            num_te_batch = 10000 // batch_size
            return teX / 255., teY, num_te_batch
    else:
        raise ValueError("Invalid dataset")


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


def find_class_by_name(modules, name):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)
