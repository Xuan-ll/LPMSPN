import os

import imageio.v2 as imageio
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform

import torch

from utils.osutils import *
from utils.imutils import *
from utils.transforms import *
import torch.utils.data as data
import torchvision.transforms as TF


class TrainingData(data.Dataset):
    def __init__(self, cfg, cfg_LL, train=True):
        self.img_folder = cfg.img_path
        self.img_folder_LL = cfg_LL.img_path
        self.is_train = train
        self.inp_res = cfg.data_shape
        self.out_res = cfg.output_shape
        self.pixel_means = cfg.pixel_means
        self.num_class = cfg.num_class
        self.cfg = cfg
        self.cfg_LL = cfg_LL
        self.bbox_extend_factor = cfg.bbox_extend_factor
        if train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry
        with open(cfg.gt_path) as anno_file:
            self.anno = json.load(anno_file)
        self.file_pairs = self.wl_ll_path_pairs()

    def augmentationCropImage(self, img, img_LL, bbox, joints=None):
        height, width = self.inp_res[0], self.inp_res[1]
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        mean_value = self.pixel_means
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
        bimg_LL = cv2.copyMakeBorder(img_LL, add, add, add, add, borderType=cv2.BORDER_CONSTANT,
                                     value=mean_value.tolist())

        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add
        if self.is_train:
            joints[:, :2] += add
            inds = np.where(joints[:, -1] == 0)
            joints[inds, :2] = -1000000  # avoid influencing by data processing
        crop_width = (bbox[2] - bbox[0]) * (1 + self.bbox_extend_factor[0] * 2)
        crop_height = (bbox[3] - bbox[1]) * (1 + self.bbox_extend_factor[1] * 2)
        if self.is_train:
            crop_width = crop_width * (1 + 0.25)
            crop_height = crop_height * (1 + 0.25)
        if crop_height / height > crop_width / width:
            crop_size = crop_height
            min_shape = height
        else:
            crop_size = crop_width
            min_shape = width

        crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
        crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
        crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

        min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
        max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
        min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
        max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

        x_ratio = float(width) / (max_x - min_x)
        y_ratio = float(height) / (max_y - min_y)

        if self.is_train:
            joints[:, 0] = joints[:, 0] - min_x
            joints[:, 1] = joints[:, 1] - min_y

            joints[:, 0] *= x_ratio
            joints[:, 1] *= y_ratio
            label = joints[:, :2].copy()
            valid = joints[:, 2].copy()

        img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
        img_LL = cv2.resize(bimg_LL[min_y:max_y, min_x:max_x, :], (width, height))
        details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype("float32")

        if self.is_train:
            return img, img_LL, joints, details
        else:
            return img, img_LL, details

    def wl_ll_path_pairs(self):
        # pairs_file_path = "/Annotations/ExLPose/brightPath2darkPath.txt"
        pairs_file_path = "../../Annotations/ExLPose/brightPath2darkPath.txt"
        wl_ll_pairs = {}
        with open(pairs_file_path) as f:
            x = f.readlines()
            for i in x:
                t = i.split(" ")
                # t[1].strip("\n")
                wl_ll_pairs[t[0]] = t[1].rstrip("\n")
        return wl_ll_pairs

    def data_augmentation(self, img, img_LL, label, operation):
        height, width = img.shape[0], img.shape[1]
        center = (width / 2., height / 2.)
        n = label.shape[0]
        affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])

        halfl_w = min(width - center[0], (width - center[0]) / 1.25 * affrat)
        halfl_h = min(height - center[1], (height - center[1]) / 1.25 * affrat)
        img = skimage.transform.resize(img[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                                       int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        img_LL = skimage.transform.resize(img_LL[int(center[1] - halfl_h): int(center[1] + halfl_h + 1),
                                          int(center[0] - halfl_w): int(center[0] + halfl_w + 1)], (height, width))
        for i in range(n):
            label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
            label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
            label[i][2] *= (
                    (label[i][0] >= 0) & (label[i][0] < width) & (label[i][1] >= 0) & (label[i][1] < height))

        return img, img_LL, label

    def __getitem__(self, index):
        a = self.anno[index]
        image_name = a['imgInfo']['img_paths']

        img_path = os.path.join(self.img_folder, image_name)
        # print(self.img_folder)
        # print(img_path)

        img_path_LL = os.path.join(self.img_folder, self.file_pairs[image_name])

        if self.is_train:
            points = np.array(a['unit']['keypoints']).reshape(self.num_class, 3).astype(np.float32)
            points_LL = np.array(a['unit']['keypoints']).reshape(self.num_class, 3).astype(np.float32)
        gt_bbox = a['unit']['GT_bbox']

        # image = scipy.misc.imread(img_path, mode='RGB')
        # image_LL = scipy.misc.imread(img_path_LL, mode='RGB')

        image = imageio.imread(img_path)
        # print(type(image))
        image_LL = imageio.imread(img_path_LL)

        rgb_mean_LL = np.mean(image_LL, axis=(0, 1))
        scaling_LL = 255 * 0.4 / rgb_mean_LL
        image_LL = image_LL * scaling_LL

        if self.is_train:
            image, image_LL, points, details = self.augmentationCropImage(image, image_LL, gt_bbox, points)
        else:
            image, image_LL, details = self.augmentationCropImage(image, image_LL, gt_bbox)

        if self.is_train:
            image, image_LL, points = self.data_augmentation(image, image_LL, points, a['operation'])
            img = im_to_torch(image)  # CxHxW
            img_LL = im_to_torch(image_LL)

            points[:, :2] //= 4  # output size is 1/4 input size
            pts = torch.Tensor(points)
        else:
            img = im_to_torch(image)
            img_LL = im_to_torch(image_LL)

        img = TF.Normalize((0.3457, 0.3460, 0.3463), (0.1477, 0.1482, 0.1483))(img)


#         if self.is_train:
#             targets = list()
#             for index in range(self.cfg.num_stage):
#                 target0 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
#                 target1 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
#                 target2 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
#                 target3 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
#                 for i in range(self.num_class):
#                     if pts[i, 2] > 0:  # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
#                         target0[i] = generate_heatmap(target0[i], pts[i], self.cfg.gk[index][0])
#                         target1[i] = generate_heatmap(target1[i], pts[i], self.cfg.gk[index][1])
#                         target2[i] = generate_heatmap(target2[i], pts[i], self.cfg.gk[index][2])
#                         target3[i] = generate_heatmap(target3[i], pts[i], self.cfg.gk[index][3])
#
#                 targets_stage = [torch.Tensor(target0), torch.Tensor(target1), torch.Tensor(target2), torch.Tensor(target3)]
#                 targets.append(targets_stage)
#
#             valid = pts[:, 2]
        if self.is_train:
            target15 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target11 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target9 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            target7 = np.zeros((self.num_class, self.out_res[0], self.out_res[1]))
            for i in range(self.num_class):
                if pts[i, 2] > 0:  # COCO visible: 0-no label, 1-label + invisible, 2-label + visible
                    target15[i] = generate_heatmap(target15[i], pts[i], self.cfg.gk15)
                    target11[i] = generate_heatmap(target11[i], pts[i], self.cfg.gk11)
                    target9[i] = generate_heatmap(target9[i], pts[i], self.cfg.gk9)
                    target7[i] = generate_heatmap(target7[i], pts[i], self.cfg.gk7)

            targets = [torch.Tensor(target15), torch.Tensor(target11), torch.Tensor(target9), torch.Tensor(target7)]
            valid = pts[:, 2]
        meta = {'index': index, 'imgID': a['imgInfo']['imgID'],
                'GT_bbox': np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]),
                'img_path': img_path, 'augmentation_details': details}

        if self.is_train:
            return img_LL, img, targets, valid, meta
        else:
            meta['det_scores'] = a['score']
            return img, meta

    def __len__(self):
        return len(self.anno)

