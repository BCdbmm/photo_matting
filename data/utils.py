# _*_ coding: utf-8 _*_
# @Time : 2020/6/19 上午10:34
# @Author : sunyulong
# @File : utils.py
import numpy as np
import cv2
import os


def get_alpha(img):
    if img.shape[2] > 3:
        alpha = img[:, :, 3]
    else:
        rdimg = np.sum(np.abs(255 - img), axis=2)
        alpha = np.where(rdimg > 100, 255, 0)
    alpha = alpha.astype(np.uint8)
    return alpha


def get_mask(alpha, thresh=50):
    return np.where(alpha > thresh, 1, 0).astype(np.uint8)

def get_trimap(alpha):
    fg=np.array(np.equal(alpha,255).astype(np.float32))
    kernal=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    iteration=np.random.randint(1,20)
    unknown=np.array(np.not_equal(alpha,0).astype(np.float))
    unknown=cv2.dilate(unknown,kernal,iterations=iteration)
    trimap=fg*255+(unknown-fg)*128
    return trimap.astype(np.uint8)

def remove_noise(gray, area_thresh=5000):
    gray = gray.astype(np.uint8)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    remove_contours = []
    for contour in contours:
        area = cv2.contourArea(contours)
        if area < area_thresh:
            remove_contours.append(contour)
    cv2.fillPoly(gray, remove_contours, 0)
    return gray


def provide(txt_path, delimiter='@'):
    """
    获取图片地址
    :param txt_path: 记录图片地址的文本文件
    :param delimiter: 分隔符
    :return: 图片路径的list
    """
    if not os.path.exists(txt_path):
        raise ValueError('txt_path does not exists')
    with open(txt_path,'r')as reader:
        content=np.load(reader,str,delimiter=delimiter)
    np.random.shuffle(content)
    image_paths=[]
    for line in content:
        paths=[x for x in line]
        image_paths.append(paths)
    return image_paths
