from Queue import Queue
# import threading
import os
from random import uniform
from random import shuffle
from random import seed
from scipy import misc
from pix_lab.util.util import read_image_list
from pix_lab.util.util import affine_transform
from pix_lab.util.util import elastic_transform
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import binary_dilation
# import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DEFAULT_MASK_EXTENSION = '.png'

class Data_provider_la(object):

    def __init__(self, path_list_train, path_list_val, n_classes, threadNum=1, queueCapacity=4, kwargs_dat={}):
        self.n_classes = n_classes
        self.list_train = None
        self.size_train = 0
        self.q_train = None
        self.list_val = None
        self.size_val = 0
        self.q_val = None

        self.mvn = kwargs_dat.get("mvn", False)
        self.seed = kwargs_dat.get("seed", 13)
        seed(self.seed)
        
        self.batchsize_tr = kwargs_dat.get("batchsize_tr", 1)
        self.batchsize_val = kwargs_dat.get("batchsize_val", 1)
        self.affine_tr = kwargs_dat.get("affine_tr", False)
        self.affine_val = kwargs_dat.get("affine_val", False)
        self.affine_value = kwargs_dat.get("affine_value", 0.025)
        self.elastic_tr = kwargs_dat.get("elastic_tr", False)
        self.elastic_val = kwargs_dat.get("elastic_val", False)
        self.elastic_value_x = kwargs_dat.get("elastic_val_x", 0.0002)
        self.elastic_value_y = kwargs_dat.get("elastic_value_y", 0.0002)
        self.rotate_tr = kwargs_dat.get("rotate_tr", False)
        self.rotate_val = kwargs_dat.get("rotate_val", False)
        self.rotateMod90_tr = kwargs_dat.get("rotateMod90_tr", False)
        self.rotateMod90_val = kwargs_dat.get("rotateMod90_val", False)
        self.skelet = kwargs_dat.get("skelet", True)
        self.dilate_num = kwargs_dat.get("dilate_num", 1)
        self.scale_min = kwargs_dat.get("scale_min", 1.0)
        self.max_val = kwargs_dat.get("max_val", None)

        self.scale_max = kwargs_dat.get("scale_max", 1.0)
        self.scale_val = kwargs_dat.get("scale_val", 1.0)
        self.one_hot_encoding = kwargs_dat.get("one_hot_encoding", True)
        self.dominating_channel = kwargs_dat.get("dominating_channel", 0)
        self.dominating_channel = min(self.dominating_channel, n_classes-1)
        self.shuffle = kwargs_dat.get("shuffle", True)

        self.queueCapacity = queueCapacity
        if path_list_train != None:
            self.list_train = read_image_list(path_list_train)
            self.size_train = len(self.list_train)
            self.q_train = self._get_list_queue(
                self.list_train,
                # self.threadNum,
                self.queueCapacity,
                # self.stopTrain,
                self.batchsize_tr,
                self.scale_min,
                self.scale_max,
                self.affine_tr,
                self.elastic_tr,
                self.rotate_tr,
                self.rotateMod90_tr
            )
        if path_list_val != None:
            self.list_val = read_image_list(path_list_val)
            if (self.max_val is not None) and (self.max_val < len(self.list_val)):
                shuffle(self.list_val) 
                self.list_val = list(sorted(self.list_val[:self.max_val]))
            self.size_val = len(self.list_val)
            self.q_val = self._get_list_queue(
                self.list_val,
                # self.threadNum,
                self.queueCapacity,
                # self.stopVal,
                self.batchsize_val,
                self.scale_val,
                self.scale_val,
                self.affine_val,
                self.elastic_val,
                self.rotate_val,
                self.rotateMod90_val
            )

#     def stop_all(self):
#         self.stopTrain.set()
#         self.stopVal.set()

#     def restart_val_runner(self):
#         if self.list_val != None:
#             self.stopVal.set()
#             self.stopVal = threading.Event()
#             self.q_val, self.threads_val = self._get_list_queue(self.list_val, self.threadNum, self.queueCapacity, self.stopVal, self.batchsize_val, self.scale_val, self.scale_val, self.affine_val, self.elastic_val, self.rotate_val, self.rotateMod90_val)

    def next_data(self, list):
        if list is 'val':
            q = self.q_val
        else:
            q = self.q_train
        if q is None:
            return None, None
        print("Val Q size: " + str(self.q_val.qsize()))
        print("Train Q size: " + str(self.q_train.qsize()))
        return q.get()


    def _get_list_queue(
        self,
        aList,
        # threadNum,
        queueCapacity,
        #stopEvent,
        batch_size,
        min_scale,
        max_scale,
        affine,
        elastic,
        rotate,
        rotateMod90
    ):
        q = Queue(maxsize=queueCapacity)
#         threads = []
#         for t in range(threadNum):
#             threads.append(
#                 threading.Thread(
#                     target=
        self._fillQueue(
            q,
            aList[:], 
    #         stopEvent,
            batch_size,
            min_scale,
            max_scale,
            affine,
            elastic,
            rotate,
            rotateMod90,
        )
        return q
#                 )
#             )
#         for t in threads:
#             t.start()
#         return q, threads



    def _fillQueue(self, q, aList, batch_size, min_scale, max_scale, affine, elastic, rotate, rotateMod90):
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
                maxH = 0
                maxW = 0
                while len(imgs) < batch_size:
                    if aIdx == len(aList):
                        if self.shuffle:
                            shuffle(aList)
                        aIdx=0
                    path = aList[aIdx]
                    aScale = uniform(min_scale, max_scale)
                    maps = []
                    imgChannels = 0
                    tgtChannels = self.n_classes
                    aImg = misc.imread(path, 'RGB')
                    if len(aImg.shape) == 2:
                        maps.append(aImg)
                        imgChannels += 1
                    else:
                        for c in range(0, aImg.shape[2]):
                            maps.append(aImg[:,:,c])
                            imgChannels += 1
                    filename, file_extension = os.path.splitext(path)

                    # In the one hot encoding scenario don not load the clutter class GT
                    to_load = tgtChannels
                    if self.one_hot_encoding:
                        to_load = tgtChannels-1
                    for aC in range(0, to_load):
                        pathTR = filename + '_GT' + str(aC) + file_extension
                        if not os.path.exists(pathTR):
                            pathTR = filename + '_GT' + str(aC) + DEFAULT_MASK_EXTENSION
                        aTgtCH = misc.imread(pathTR, 'L')
                        # Mask might be smaller or larger than src image in case of vendor custom GT files
                        
                        if aTgtCH.shape[:2] != aImg.shape[:2]:
                            aTgtCH = misc.imresize(aTgtCH, aImg.shape[0:2], interp='bicubic')
                        maps.append(aTgtCH)
                    resizedMaps = []
                    for c in range(0, imgChannels + to_load):
                        resizedMaps.append(
                            np.expand_dims(
                                misc.imresize(maps[c], aScale, interp='bicubic'),
                                2))
                    res = np.dstack(resizedMaps)

                    if affine:
                        res = affine_transform(res, self.affine_value)
                    if elastic:
                        res = elastic_transform(res, self.elastic_value_x, self.elastic_value_y)
                    if rotate or rotateMod90:
                        angle = uniform(0, 360)
                        if rotateMod90:
                            if angle < 90:
                                angle = 0.0
                            elif angle < 180.0:
                                angle = 90.0
                            elif angle < 270.0:
                                angle = 180.0
                            else:
                                angle = 270.0
                        res = ndimage.interpolation.rotate(res, angle)
                    aImg = res[:, :, 0:imgChannels]
                    aTgt = res[:, :, imgChannels:]

                    aTgt = np.where(aTgt > 64, 1.0, 0.0)
                    if self.skelet:
                        for c in range(0,tgtChannels-1):
                            tTgt = skeletonize(aTgt[:,:,c])
                            tTgt = binary_dilation(tTgt,iterations=self.dilate_num)
                            aTgt[:, :, c] = tTgt
                    # Ensure a one-hot-encoding (could be damaged due to transformations)
                    # First Channel is of highest importance
                    if self.one_hot_encoding:
                        aMap = aTgt[:, :, self.dominating_channel]
                        for aM in range(0, tgtChannels - 1):
                            if aM == self.dominating_channel:
                                continue
                            else:
                                tMap = np.logical_and(aTgt[:, :, aM], np.logical_not(aMap))
                                aMap = np.logical_or(aMap, tMap)
                                aTgt[:, :, aM] = tMap
                        # Add+Calculate the clutter map
                        aTgt = np.pad(aTgt, ((0,0),(0,0),(0,1)), mode='constant')
                        aTgt[:, :, tgtChannels - 1] = np.logical_not(aMap)
                    aImg = aImg / 255.0
                    imgs.append(aImg)
                    width = aImg.shape[1]
                    heigth = aImg.shape[0]
                    maxW = max(width, maxW)
                    maxH = max(heigth, maxH)
                    tgts.append(aTgt)

                for cImg in imgs:
                    heigth = cImg.shape[0]
                    padH = maxH - heigth
                    width = cImg.shape[1]
                    padW = maxW - width
                    if padH + padW > 0:
                        npad = ((0, padH), (0, padW))
                        cImg = np.pad(cImg, npad, mode='constant', constant_values=0)
                    imgsFin.append(np.expand_dims(cImg, 0))
                for cTgt in tgts:
                    heigth = cTgt.shape[0]
                    padH = maxH - heigth
                    width = cTgt.shape[1]
                    padW = maxW - width
                    aTgtContent = cTgt[:, :, 0:cTgt.shape[2] - 1]
                    aTgtBackG = cTgt[:, :, cTgt.shape[2] - 1]
                    if padW + padH > 0:
                        npad = ((0, padH), (0, padW), (0, 0))
                        aTgtContent = np.pad(aTgtContent, npad, mode='constant', constant_values=0)
                        aTgtBackG = np.pad(aTgtBackG, npad, mode='constant', constant_values=1)
                    tgtsFin.append(np.expand_dims(np.dstack([aTgtContent, aTgtBackG]), 0))
                bX = np.concatenate(imgsFin)
                bT = np.concatenate(tgtsFin)
                curPair = [bX, bT]
            try:
                print('Putting to queue', aIdx)
                q.put(curPair)
                curPair = None
                aIdx += 1
            except:
                continue