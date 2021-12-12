from multiprocessing import Queue
import threading
import os
import numpy
from scipy import ndimage
from PIL import Image

import random
# import uniform
# from random import shuffle
# from random import seed
from scipy import misc
from .util import read_image_list
from .util import atransform
from .util import elastic_transform
from skimage import morphology

DEBUG = bool(os.environ.get('DEBUG', False))

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MASK_EXTENSION = '.png'

def scale_to_area(width, height, target_area):
    ratio = height / width

    a = (target_area / ratio) ** (1 / 2)

    width_new = int(a)
    height_new = int(a * ratio)
    return width_new, height_new

def load_img(
    path,
    debug,
    channels,
    max_pixels,
    n_classes,
    one_hot_encoding,
    elastic_value_x,
    elastic_value_y,
    elastic,
    affine_value,
    affine,
    rotate,
    rotateMod90,
    skelet,
    dilate_num,
    dominating_channel,
    min_scale,
    max_scale,
):
    # path = aList[aIdx]
    # Align with max pixels constraint
    maps = []

    if debug:
        print('Data provider loading:', path)

    with Image.open(path) as im:
        if channels == 1:
            aImg = im.convert('L')
        else:
            aImg = im.convert('RGB')

    # constrained_w, constrained_h  = downscale_dims(aImg.width, aImg.height)
    constrained_w, constrained_h  = scale_to_area(aImg.width, aImg.height, max_pixels)
    constrained_scale = max(constrained_w / aImg.width, constrained_h / aImg.height)
    max_scale = min(constrained_scale, max_scale)
    min_scale = min(min_scale, max_scale)
    aScale = random.uniform(min_scale, max_scale)

    maps.append(aImg)

    # In the one hot encoding scenario don not load the clutter class GT
    to_load = n_classes
    if one_hot_encoding:
        to_load = n_classes - 1

    filename, file_extension = os.path.splitext(path)
    for aC in range(0, to_load):
        pathTR = '{}_GT{}{}'.format(filename, aC, file_extension)
        if not os.path.exists(pathTR):
            pathTR = '{}_GT{}{}'.format(filename, aC, DEFAULT_MASK_EXTENSION)
        if debug:
            print('Data provider loading:', pathTR)

        with Image.open(pathTR) as im:
            aTgtCH = im.convert('L')

        # Mask might be smaller or larger than src image in case of vendor custom GT files

        if aTgtCH.width != aImg.width:
            aTgtCH = aTgtCH.resize((aImg.width, aImg.height), Image.LANCZOS)
        maps.append(aTgtCH)

    if debug:
        print('maps', [(m.width, m.height,) for m in maps])

    resized_maps = []
    for c in range(0, channels + to_load):
        imm = maps[c]
        resized_maps.append(
            numpy.expand_dims(
                numpy.array(imm.resize(
                    (
                        int(imm.width * aScale),
                        int(imm.height * aScale)
                    ),
                    Image.LANCZOS,
                )),
                2
            )
        )
    res = numpy.dstack(resized_maps)
    if debug:
        print('Loaded resized maps shape:', res.shape)

    if affine:
        res = atransform(res, affine_value)
    if elastic:
        res = elastic_transform(res, elastic_value_x, elastic_value_y)
    if rotate or rotateMod90:
        angle = random.randrange(4) * 90.0 if rotateMod90 else random.uniform(0, 360)
        res = ndimage.interpolation.rotate(res, angle)
    aImg = res[:, :, 0:channels]
    aTgt = res[:, :, channels:]


    aTgt = numpy.where(aTgt > 64, 1.0, 0.0)
    if skelet:
        for c in range(0, n_classes-1):
            tTgt = morphology.skeletonize(aTgt[:,:,c])
            tTgt = morphology.binary_dilation(tTgt, iterations=dilate_num)
            aTgt[:, :, c] = tTgt
    # Ensure a one-hot-encoding (could be damaged due to transformations)
    # First Channel is of highest importance
    if one_hot_encoding:
        aMap = aTgt[:, :, dominating_channel]
        for aM in range(0, n_classes - 1):
            if aM == dominating_channel:
                continue
            else:
                tMap = numpy.logical_and(aTgt[:, :, aM], numpy.logical_not(aMap))
                aMap = numpy.logical_or(aMap, tMap)
                aTgt[:, :, aM] = tMap
        # Add+Calculate the clutter map
        aTgt = numpy.pad(aTgt, ((0,0),(0,0),(0,1)), mode='constant')
        aTgt[:, :, n_classes - 1] = numpy.logical_not(aMap)
    aImg = aImg / 255.0
    return aImg, aTgt
        
class Data_provider_la(object):
    """
    Image pre-processing: input image I is 
    - 1/2 for max{Ih, Iw} < 2000, 
    - 1/3 for 2000 ≤ max{Ih, Iw} < 4800 
    - 1/4 for >=4800
    followed by a normalization to mean 0 and variance 1 (on pixel intensity level)
    Source: https://arxiv.org/pdf/1802.03345.pdf
    
    Preprocessing strategies:

    1.  subsampled by a constant factor of 3 (no further data augmentation - one training sample per element of the training set) – B
    
    2.  randomly subsampled by a factor s ∈ [2, 5] – S
    
    3.  S + random affine transformation (three corner points of the image are randomly shifted within a circle
        with D = 0.025 · max(Ih, Iw) (around there original position) – S + A

    4.  S + A + elastic transformation – S + A + E
    """
    def __init__(self, path_list_train, path_list_val=None, n_classes=None, thread_num=1, queue_capacity=4, kwargs_dat={}):
        self.n_classes = n_classes
        self.list_train = None
        self.size_train = 0
        self.q_train = None
        self.list_val = None
        self.size_val = 0
        self.q_val = None

        self.mvn = kwargs_dat.get("mvn", False)
        self.seed = kwargs_dat.get("seed", 13)
        random.seed(self.seed)
        
        self.batchsize_tr = kwargs_dat.get("batchsize_tr", 1)
        self.batchsize_val = kwargs_dat.get("batchsize_val", 1)
        self.affine_tr = kwargs_dat.get("affine_tr", False)
        self.affine_val = kwargs_dat.get("affine_val", False)
        self.affine_value = kwargs_dat.get("affine_value", 0.025)
        self.elastic_tr = kwargs_dat.get("elastic_tr", False)
        self.elastic_val = kwargs_dat.get("elastic_val", False)
        self.elastic_value_x = kwargs_dat.get("elastic_val_x", 0.0002)
        self.elastic_value_y = kwargs_dat.get("elastic_value_y", 0.0002)
        self.rotate = kwargs_dat.get("rotate_tr", False)
        self.rotate = kwargs_dat.get("rotate_val", False)
        self.rotateMod90_tr = kwargs_dat.get("rotateMod90_tr", False)
        self.rotateMod90_val = kwargs_dat.get("rotateMod90_val", False)
        self.skelet = kwargs_dat.get("skelet", True)
        self.dilate_num = kwargs_dat.get("dilate_num", 1)
        self.scale_min = kwargs_dat.get("scale_min", 1.0)
        self.max_val = kwargs_dat.get("max_val", None)
        self.max_pixels = kwargs_dat.get("max_pixels", 4 * 1024 * 1024)
        self.scale_max = kwargs_dat.get("scale_max", 1.0)
        self.one_hot_encoding = kwargs_dat.get("one_hot_encoding", True)
        self.val_mode = kwargs_dat.get("val_mode", False)

        self.dominating_channel = kwargs_dat.get("dominating_channel", 0)
        self.dominating_channel = min(self.dominating_channel, n_classes - 1)
        self.shuffle = kwargs_dat.get("shuffle", True)
        self.debug = kwargs_dat.get("debug", False) or DEBUG
        self.channels = kwargs_dat.get("channels", 3)
        self.thread_num = thread_num
        self.queue_capacity = queue_capacity
        self.stopTrain = threading.Event()
        self.stopVal = threading.Event()
        if path_list_train != None:
            self.list_train = read_image_list(path_list_train)
            self.size_train = len(self.list_train)
            self.q_train, self.threads_tr = self._get_list_queue(self.list_train, self.thread_num, self.queue_capacity, self.stopTrain, self.batchsize_tr,
                                                self.scale_min, self.scale_max, self.affine_tr, self.elastic_tr, self.rotate, self.rotateMod90_tr)
#         if path_list_val != None:
#             self.list_val = read_image_list(path_list_val)
#             if (self.max_val is not None) and (self.max_val < len(self.list_val)):
#                 shuffle(self.list_val) 
#                 self.list_val = list(sorted(self.list_val[:self.max_val]))
#             self.size_val = len(self.list_val)
#             self.q_val, self.threads_val = self._get_list_queue(self.list_val, self.thread_num, self.queue_capacity, self.stopVal, self.batchsize_val, self.scale_min, self.scale_max, self.affine_val, self.elastic_val, self.rotate, self.rotateMod90_val)

    def stop_all(self):
        self.stopTrain.set()
        if self.debug:
            print('[Data provider] Train set stopped')
        self.stopVal.set()
        if self.debug:
            print('[Data provider] Val set stopped')

    def restart_val_runner(self):
        if self.list_val != None:
            self.stopVal.set()
            self.stopVal = threading.Event()
            self.q_val, self.threads_val = self._get_list_queue(self.list_val, self.thread_num, self.queue_capacity, self.stopVal, self.batchsize_val, self.scale_min, self.scale_max, self.affine_val, self.elastic_val, self.rotate, self.rotateMod90_val)

    def next_data(self, list=None):
        if self.q_train is None:
            return None
        if self.debug:
            print("Train Q size: " + str(self.q_train.qsize()))
        return self.q_train.get()


    def _get_list_queue(self, aList, thread_num, queue_capacity, stopEvent, batch_size, min_scale, max_scale, affine, elastic, rotate, rotateMod90):
        return Queue(maxsize=queue_capacity)
#         threads = []
#         for t in range(thread_num):
#             threads.append(
#                 threading.Thread(
#                     target=self._fillQueue,
#                     daemon=True,
#                     args=(
#                         q,
#                         aList[:], 
#                         stopEvent,
#                         batch_size,
#                         min_scale,
#                         max_scale,
#                         affine,
#                         elastic,
#                         rotate,
#                         rotateMod90,
#                     )
#                 )
#             )
#         for t in threads:
#             t.start()
#         return q, threads


    def _fillQueue(self, q, aList, stopEvent, batch_size, min_scale, max_scale, affine, elastic, rotate, rotateMod90):
        if self.shuffle:
            shuffle(aList)
        aIdx = 0
        curPair = None
        while (not stopEvent.is_set()):
            if curPair is None:
                imgs = []
                imgsFin = []
                tgts = []
                tgtsFin = []
                paths = []
                maxH = 0
                maxW = 0
                while len(imgs) < batch_size:
                    if aIdx == len(aList):
                        if self.shuffle:
                            shuffle(aList)
                        aIdx=0
                    aImg, aTgt = load_img(
                        path=aList[aIdx],
                        debug=self.debug,
                        channels=self.channels,
                        max_pixels=self.max_pixels,
                        n_classes=self.n_classes,
                        one_hot_encoding=self.one_hot_encoding,
                        elastic_value_x=self.elastic_value_x,
                        elastic_value_y=self.elastic_value_y,
                        elastic=elastic,
                        affine_value=self.affine_value,
                        affine=affine,
                        rotate=rotate,
                        rotateMod90=rotateMod90,
                        skelet=self.skelet,
                        dilate_num=self.dilate_num,
                        dominating_channel=self.dominating_channel,
                        min_scale=min_scale,
                        max_scale=max_scale,
                    )
                    width = aImg.shape[1]
                    heigth = aImg.shape[0]
                    maxW = max(width, maxW)
                    maxH = max(heigth, maxH)
                    imgs.append(aImg)
                    tgts.append(aTgt)
                    paths.append(aList[aIdx])


                for cImg in imgs:
                    heigth = cImg.shape[0]
                    padH = maxH - heigth
                    width = cImg.shape[1]
                    padW = maxW - width
                    if padH + padW > 0:
                        npad = ((0, padH), (0, padW))
                        cImg = numpy.pad(cImg, npad, mode='constant', constant_values=0)
                    imgsFin.append(numpy.expand_dims(cImg, 0))
                for cTgt in tgts:
                    heigth = cTgt.shape[0]
                    padH = maxH - heigth
                    width = cTgt.shape[1]
                    padW = maxW - width
                    aTgtContent = cTgt[:, :, 0:cTgt.shape[2] - 1]
                    aTgtBackG = cTgt[:, :, cTgt.shape[2] - 1]
                    if padW + padH > 0:
                        npad = ((0, padH), (0, padW), (0, 0))
                        aTgtContent = numpy.pad(aTgtContent, npad, mode='constant', constant_values=0)
                        aTgtBackG = numpy.pad(aTgtBackG, npad, mode='constant', constant_values=1)
                    tgtsFin.append(numpy.expand_dims(numpy.dstack([aTgtContent, aTgtBackG]), 0))
                bX = numpy.concatenate(imgsFin)
                bT = numpy.concatenate(tgtsFin)
                curPair = [bX, bT]
            try:
                if self.debug:
                    print('Enqueuing', aIdx, type(bX), type(bT))
                if self.val_mode:
                    q.put(curPair + [paths])
                else:
                    q.put(curPair)
                curPair = None
                aIdx += 1
                if self.debug:
                    print('Puttong to QUEUE done, aIdx', aIdx)
            except Exception as e:
                print('Puttong to QEUE failed', e)
                continue
