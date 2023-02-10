import math
import cv2
import numpy as np

class ResizeImg(object):
    def __init__(self, image_shape, max_seq_len, width_downsample_ratio=0.25):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        img = data['image']
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img(
            img, self.image_shape, self.width_downsample_ratio)
        word_positons = robustscanner_other_inputs(self.max_seq_len)
        data['image'] = norm_img
        data['resized_shape'] = resize_shape
        data['pad_shape'] = pad_shape
        data['valid_ratio'] = valid_ratio
        data['word_positons'] = word_positons
        return data

def resize_norm_img(img, image_shape, width_downsample_ratio=0.25):
    imgC, imgH, imgW_min, imgW_max = image_shape
    h = img.shape[0]
    w = img.shape[1]
    valid_ratio = 1.0
    # make sure new_width is an integral multiple of width_divisor.
    width_divisor = int(1 / width_downsample_ratio)
    # resize
    ratio = w / float(h)
    resize_w = math.ceil(imgH * ratio)
    if resize_w % width_divisor != 0:
        resize_w = round(resize_w / width_divisor) * width_divisor
    if imgW_min is not None:
        resize_w = max(imgW_min, resize_w)
    if imgW_max is not None:
        valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
        resize_w = min(imgW_max, resize_w)
    resized_image = cv2.resize(img, (resize_w, imgH))
    resized_image = resized_image.astype('float32')
    # norm 
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    resize_shape = resized_image.shape
    padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
    padding_im[:, :, 0:resize_w] = resized_image
    pad_shape = padding_im.shape

    return padding_im, resize_shape, pad_shape, valid_ratio

def robustscanner_other_inputs(max_text_length):
    word_pos = np.array(range(0, max_text_length)).astype('int64')
    return word_pos