import glob
import numpy
import re
import os
import tensorflow.compat.v1 as tf
from scipy.ndimage.interpolation import affine_transform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

GT_RE = re.compile("^.+[_\-]gt[0-9]*\.[^.\/]+$", flags=re.IGNORECASE)

    
def read_image_list(pathToList):
    '''

    :param pathToList:
    :return:
    '''
    pathToList = os.path.normpath(pathToList)
    if pathToList.endswith('.lst') or pathToList.endswith('.txt'):
        f = open(pathToList, 'r')
        filenames = []
        for line in f:
            if line[-1] == '\n':
                filenames.append(line[:-1])
            else:
                filenames.append(line)
        f.close()
    else:
        filenames = [
            fn 
            for fn in glob.glob(pathToList)
            if not GT_RE.match(fn)
        ]
    return list(sorted(filenames))


def calcAffineMatrix(sourcePoints, targetPoints):
    # For three or more source and target points, find the affine transformation
    A = []
    b = []
    for sp, trg in zip(sourcePoints, targetPoints):
        A.append([sp[0], 0, sp[1], 0, 1, 0])
        A.append([0, sp[0], 0, sp[1], 0, 1])
        b.append(trg[0])
        b.append(trg[1])
    result, resids, rank, s = numpy.linalg.lstsq(numpy.array(A), numpy.array(b), rcond=None)

    a0, a1, a2, a3, a4, a5 = result
    affineTrafo = numpy.float32([[a0, a2, a4], [a1, a3, a5]])
    return affineTrafo

def atransform(image, affine_value):
    shape = image.shape
    alpha_affine = min(shape[0], shape[1]) * affine_value
    random_state = numpy.random.RandomState(None)
    # Random affine
    shape_size = shape[:2]
    center_square = numpy.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = numpy.float32(
        [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
         center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(numpy.float32)
    M = calcAffineMatrix(pts1, pts2)
    R = M[0:2, 0:2]
    Off = M[:, 2]
    for aD in range(shape[2]):
        image[:, :, aD] = affine_transform(image[:, :, aD], R, offset=Off)
    return image

def elastic_transform(image,elastic_value_x ,elastic_value_y):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications JUST in Y-DIRECTION).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    shape = image.shape
    random_state = numpy.random.RandomState(None)
    nY = shape[0] // 25
    nX = shape[1] // 25
    sigma = min(shape[1], shape[0]) * 0.0025
    alpha_X = elastic_value_x * min(shape[0], shape[1])
    alpha_Y = elastic_value_y * min(shape[0], shape[1])
    dx = gaussian_filter((random_state.rand(nY, nX) * 2 - 1), sigma)
    dy = gaussian_filter((random_state.rand(nY, nX) * 2 - 1), sigma)
    x, y, z = numpy.meshgrid(numpy.arange(shape[1]), numpy.arange(shape[0]), numpy.arange(shape[2]))
    dx = numpy.array(Image.from_array(dx).resize(shape[0:2], Image.LANCZOS))
    dy = numpy.array(Image.from_array(dy).resize(shape[0:2], Image.LANCZOS))
    # plt.imshow(dx, cmap=plt.cm.gray)
    # plt.show()
    dxT = []
    dyT = []
    for dummy in range(shape[2]):
        dxT.append(dx)
        dyT.append(dy)
    dx = numpy.dstack(dxT)
    dy = numpy.dstack(dyT)
    dx = dx * alpha_X
    dy = dy * alpha_Y
    indices = numpy.reshape(y + dy, (-1, 1)), numpy.reshape(x + dx, (-1, 1)), numpy.reshape(z, (-1, 1))
    image = map_coordinates(image, indices, order=1).reshape(shape)
    return image

