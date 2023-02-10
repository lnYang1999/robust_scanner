import os
import cv2
import numpy as np

import mindspore as ms
import mindspore.dataset as ds

dataset_dic={'Icdar2013','Icdar2015','Icdar2017'}

class IcdarDataset():
    def __init__(self, dataset_tag=None, gt_root=None, img_root=None, 
                 batchsize=1, training=True,
                 img_w=160, img_h=48, case_sensitive=True,
                 testing_with_label_file=False, convert_to_gray=True):
        
        assert dataset_tag is not None, 'dataset_tag must be set'
        if dataset_tag not in dataset_dic:
            assert 'dataset_tag not correct' 
        assert gt_root is not None, 'gt root must be set'
        assert img_root is not None, 'img root must be set'

        self.all_images = []
        self.all_labels = []
        self.all_bboxes = []

        self.img_w = img_w
        self.img_h = img_h
        self.batchsize = batchsize

        self.training = training
        self.case_sensitive = case_sensitive
        self.testing_with_label_file = testing_with_label_file
        self.convert_to_gray = convert_to_gray

        im_name_list = os.listdir(img_root)
        gt_name_list = ["gt_" + name.split('.')[0] + ".txt" for name in im_name_list]

        for im_name in zip(im_name_list):
            im = cv2.imread(os.path.join(img_root, im_name))
            self.all_images.append(im)

        ann = []
        if dataset_tag == 'Icdar2013':
            ann = get_icdar_2013_info(gt_root, gt_name_list)
            self.all_labels.append(ann['labels'])
            self.all_bboxes.append(ann['bboxes'])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img = self.all_images[idx]
        label = self.all_labels[idx]
        bbox = self.all_bboxes[idx]
        return (img, label, bbox)
        

def get_icdar_2013_info(gt_dir, gt_name_list):
    gt_bboxes = []
    gt_labels = []
    for gt_name in zip(gt_name_list):
        bboxes = []
        labels = []
        gt_list = list()
        with open(os.path.join(gt_dir, gt_name)) as f:
            gt_list = f.readlines()
        gt_list = [gt.strip().split(',') for gt in gt_list]
        for gt in gt_list:
            x1, y1, x2, y2 = gt[:4]  # 拿到两个点
            label = ''.join(gt[4:])
            label = label.replace('"', '')
            print(label)  ## label
            bbox = [x1, y1, x2, y2]
            bboxes.append(bbox)
            labels.append(label)
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels)
    return ann

    

class ICDAR2015Dataset():
    def __init__(self, n):
        self.data = []
        self.label = []
        for _ in range(n):
            self.data.append(np.zeros((3, 4, 5)))
            self.label.append(np.ones((1)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

if __name__ == "__main__":
    gt_root = '/home/data/ysp22/dataset/ICDAR2013/Challenge2_Training_Task1_GT'
    img_root = '/home/data/ysp22/dataset/ICDAR2013/Challenge2_Training_Task12_Images'
    icdar_dataset = IcdarDataset(dataset_tag='Icdar2013', gt_root=gt_root, img_root=img_root)
    dataset = ds.GeneratorDataset(icdar_dataset,["img","labels","bboxes"],shuffle=False)
