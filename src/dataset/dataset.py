# -*- coding: utf-8 -*-
import os
import random
import io
import math
import json

import cv2
from PIL import Image
import numpy as np
import lmdb

import mindspore as ms
import mindspore.dataset as ds

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import sampler
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from utils.GeometryUtils import resize_with_height, pad_image_with_specific_base
from utils.label_util import LabelTransformer


# mindspore版
class TextDataset():

    def __init__(self, txt_file=None, img_root=None, transform=None, target_transform=None, training=True,
                 img_w=256, img_h=32, case_sensitive=True,
                 testing_with_label_file=False, convert_to_gray=True, split=','):
        '''

        :param txt_file: txt file, every line containing <ImageFile>,<Text Label>
        :param img_root:
        :param transform:
        :param target_transform:....
        :param training:
        :param img_w:
        :param img_h:
        :param testing_with_label_file: if False, read image from img_root, otherwise read img from txt_file
        '''
        assert img_root is not None, 'root must be set'
        self.img_w = img_w
        self.img_h = img_h

        self.training = training
        self.case_sensitive = case_sensitive
        self.testing_with_label_file = testing_with_label_file
        self.convert_to_gray = convert_to_gray

        self.all_images = []
        self.all_labels = []

        if training or testing_with_label_file:  # for train and validation
            images, labels = get_datasets_image_label_with_txt_file(txt_file, img_root, split)
            self.all_images += images
            self.all_labels += labels
            # print("all_images:",len(self.all_images))
            # print("all_labels:",len(self.all_labels))
        else:  # for testing, root is image_dir
            imgs = os.listdir(img_root)
            for img in imgs:
                self.all_images.append(os.path.join(img_root, img))

        # for debug
        self.all_images = self.all_images[:]
        self.nSamples = len(self.all_images)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, _index):
        try:
            file_name = self.all_images[_index]
            label = ''
            if self.training:
                label = self.all_labels[_index]

            img = Image.open(file_name)

            try:
                if self.convert_to_gray:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
            except Exception as e:
                print('Error Image for {}'.format(file_name))

            if self.transform is not None:
                img, width_ratio = self.transform(img)

            if self.target_transform is not None and self.training:
                label = self.target_transform(label)
            if self.training:
                if not self.case_sensitive:
                    label = label.lower()
                return img, label
            else:
                return img, file_name
        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))

    def get_all_labels(self):
        return self.all_labels


class JSONDataset(Dataset):
    def __init__(self, txt_file=None, img_root=None, transform=None, target_transform=None, training=True,
                 img_w=256,
                 img_h=32, case_sensitive=True, testing_with_label_file=False, convert_to_gray=True):
        '''
        :param txt_file: txt file, every line containing {ImageFile:<ImageFile>, Label:<Label>}
        :param img_root:
        :param transform:
        :param target_transform:
        :param training:
        :param img_w:
        :param img_h:
        :param testing_with_label_file: if False, read image from img_root, otherwise read img from txt_file
        '''
        assert img_root is not None, 'root must be set'
        self.img_w = img_w
        self.img_h = img_h

        self.training = training
        self.case_sensitive = case_sensitive
        self.testing_with_label_file = testing_with_label_file
        self.convert_to_gray = convert_to_gray

        self.all_images = []
        self.all_labels = []

        if training or testing_with_label_file:  # for train and validation
            images, labels = get_dataset_image_and_label_with_json_file(txt_file, img_root)
            self.all_images += images
            self.all_labels += labels
        else:  # for testing, root is image_dir
            imgs = os.listdir(img_root)
            for img in imgs:
                self.all_images.append(os.path.join(img_root, img))

        # for debug
        self.all_images = self.all_images[:]
        # print("val_dataset:",self.all_images)
        self.nSamples = len(self.all_images)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            file_name = self.all_images[index]
            if self.training:
                label = self.all_labels[index]

            img = Image.open(file_name)

            try:
                if self.convert_to_gray:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
            except Exception as e:
                print('Error Image for {}'.format(file_name))
                # exit()

            if self.transform is not None:
                img, width_ratio = self.transform(img)

            if self.target_transform is not None and self.training:
                label = self.target_transform(label)
            if self.training:
                if not self.case_sensitive:
                    label = label.lower()
                return (img, label)
            else:
                return (img, file_name)
        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))

def get_datasets_image_label_with_txt_file(txt_file, img_root, split=','):
    image_names = []
    labels = []
    # every line containing <ImageFile,Label> text
    with open(txt_file, encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            splited = line.strip().rstrip('\n').split(split)
            image_name = splited[0]
            label = split.join(splited[1:])

            #####由于txt中的字符带“”，为了与example中的保持一致，故添加
            label = label.strip('"').rstrip('"')

            if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                continue
            image_name = os.path.join(img_root, image_name)
            image_names.append(image_name)
            labels.append(label)
    return image_names, labels


def get_dataset_image_and_label_with_json_file(txt_file, img_root):
    image_names = []
    labels = []

    # every line containing {ImageFile:<ImageFile>, Label:<Label>} json object
    with open(txt_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().rstrip("\n")
            lineobj = json.loads(line)
            label = lineobj['Label']
            if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                continue
            image_name = lineobj['ImageFile']
            image_name = os.path.join(img_root, image_name)
            image_names.append(image_name)
            labels.append(label)
    return image_names, labels


class ResizeWeight(object):

    def __init__(self, size, interpolation=Image.BILINEAR, gray_format=True):
        self.w, self.h = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.gray_format = gray_format

    def __call__(self, img):
        img_w, img_h = img.size

        if self.gray_format:
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:self.h, 0] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:new_w, 0] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[:, :, 0] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w
        else:  # RGB format
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:self.h, :] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:new_w, :] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[:, :, :] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w


# mindspore版
class CustomImagePreprocess:
    def __init__(self, _target_height, _target_width, _is_gray):
        self.target_height, self.target_width = _target_height, _target_width
        self.is_gray = _is_gray

    def __call__(self, _img: Image.Image):
        if self.is_gray:
            img = _img.convert('L')
        else:
            img = _img
        img_np = np.asarray(img)
        h, w = img_np.shape[:2]
        resized_img = cv2.resize(img_np, (self.target_width, self.target_height))
        full_channel_img = resized_img[..., None] if len(resized_img.shape) == 2 else resized_img
        # resized_img_tensor = torch.from_numpy(np.transpose(full_channel_img, (2, 0, 1))).to(torch.float32)
        # resized_img_tensor.sub_(127.5).div_(127.5)
        resized_img_tensor = ms.Tensor.from_numpy(np.ascontiguousarray(np.transpose(full_channel_img, (2, 0, 1))))
        resized_img_tensor = ms.ops.sub(resized_img_tensor, 127.5)
        resized_img_tensor = ms.ops.div(resized_img_tensor, 127.5)

        return resized_img_tensor, w / self.target_width


class DistCollateFn(object):
    '''
    fix bug when len(data) do not be divided by batch_size, on condition of distributed validation
    avoid error when some gpu assigned zero samples
    '''

    def __init__(self, training=True):
        self.training = training

    def __call__(self, batch):

        batch_size = len(batch)
        if batch_size == 0:
            return dict(batch_size=batch_size, images=None, labels=None)

        if self.training:
            images, labels = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        labels=labels)
        else:
            images, file_names = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        file_names=file_names)

class DistValSampler(Sampler):
    # DistValSampler distributes batches equally (based on batch size) to every gpu (even if there aren't enough samples)
    # This instance is used as batch_sampler args of validation dtataloader,
    # to guarantee every gpu validate different samples simultaneously
    # WARNING: Some baches will contain an empty array to signify there aren't enough samples
    # distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per process. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_steps = math.ceil(len(self.indices) / self.world_size / self.batch_size)

        # num_samples = total samples / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_steps * self.batch_size

    def __iter__(self):
        current_rank_offset = self.num_samples * self.global_rank
        current_sampled_indices = self.indices[
                                  current_rank_offset:min(current_rank_offset + self.num_samples, len(self.indices))]

        for step in range(self.expected_num_steps):
            step_offset = step * self.batch_size
            # yield sampled_indices[offset:offset + self.batch_size]
            yield current_sampled_indices[step_offset:min(step_offset + self.batch_size, len(current_sampled_indices))]

    def __len__(self):
        return self.expected_num_steps

    def set_epoch(self, epoch):
        return


# mindspore版
# 通过调用get_dataset，相当于获得pytorch的dataloader
class GeneratorDataset():
    def __init__(self, source, label, batch_size, sampler=None, num_parallel_workers=1,
                 shuffle=True):
        self.dataset = ds.GeneratorDataset(source, label, sampler=sampler,
                                           num_parallel_workers=num_parallel_workers,
                                           shuffle=shuffle)
        self.dataset = self.dataset.batch(batch_size)

    def get_dataset(self):
        return self.dataset


if __name__ == "__main__":
    text_file = "/home/data/qxh22/dataset/dataset_masters/SynthText_Add_new/annotationlist/gt_1.txt"
    img_root = "/home/data/qxh22/dataset/dataset_masters/SynthText_Add_new/crop_img_1"
    img_h = 48
    img_w = 160
    convert_to_gray = True
    train_dataset = TextDataset(text_file, img_root,
                                 transform=CustomImagePreprocess(img_h, img_w, convert_to_gray),
                                 convert_to_gray=convert_to_gray)
    # print(type(train_dataset))
    # print("成功！！！")
    train_dataset = GeneratorDataset(train_dataset, ["data", "label"], batch_size=128, shuffle=True,
                                     num_parallel_workers=2).get_dataset()
    # print("成功222222222！！！")
    # print("成功333333333！！！")

    steps = train_dataset.get_dataset_size()
    class_indexing = train_dataset.get_class_indexing()
    print("setps:", steps)
    # # import cv2 as cv
    # import matplotlib.pyplot as plt
    # for data in datasett.create_dict_iterator():
    #     # print(type(data["data"]),type(data["label"]))
    #     # print(data["data"].shape, data["label"])
    #     at = data["data"].asnumpy().squeeze(axis=0)

    #     at=at*128+128
    #     at = at.astype(np.uint8)
    #     print(at)
    #     im = Image.fromarray(at)
    #     im.save("filename{}.png".format(i))
    #     i=i+1
    #     if i>=3:
    #         break



