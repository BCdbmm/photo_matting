# _*_ coding: utf-8 _*_
# @Time : 2020/6/19 下午2:44
# @Author : sunyulong
# @File : retrieve.py
import cv2
import numpy as np
import utils
import os


def get_image_paths(root_dir):
    image_paths_dict = {}
    matting_paths_dict = {}
    for root, dirs, files in os.walk(root_dir):
        if not files:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            file_path = file_path.replace('\\', '/')
            image_name = file.split('.')[0]
            dir_name = file_path.split('/')[-2]
            if dir_name.startswith('clip'):
                image_paths_dict[image_name] = file_path
            if dir_name.startswith('matting'):
                matting_paths_dict[image_name] = file_path
    image_corresponding = []
    for image_name, path in image_paths_dict.items():
        if image_name not in matting_paths_dict:
            continue
        matting_path = matting_paths_dict[image_name]
        image_corresponding.append([path, matting_path])
    if len(image_corresponding) < 1:
        print(root_dir)
        raise ValueError('root_dir is error')
    return image_corresponding


def split_images(image_paths, num_val=1000):
    if image_paths is None:
        raise ValueError('image_paths should not None')
    np.random.shuffle(image_paths)
    val_image_paths = image_paths[:num_val]
    train_image_paths = image_paths[num_val:]
    return train_image_paths, val_image_paths


def save_image_path(image_paths, txt_path, delimiter='@'):
    if image_paths is None:
        raise ValueError('image_paths should not None')
    with open(txt_path, 'w')as f:
        for el in image_paths:
            line = delimiter.join(el)
            f.write(line + '\n')


def save_masks_alphas(image_paths, root_dir, add_mask_paths=True):
    alpha_dir = os.path.join(root_dir, 'alphas')
    mask_dir = os.path.join(root_dir, 'masks')
    trimap_dir = os.path.join(root_dir, 'trimaps')
    if not os.path.exists(alpha_dir):
        os.mkdir(alpha_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    if not os.path.exists(trimap_dir):
        os.mkdir(trimap_dir)
    new_image_paths = []
    for i, [clip, matting] in enumerate(image_paths):
        matting_img = cv2.imread(matting, -1)
        if matting_img is None:
            print("{} does not exist".format(matting))
        alpha = utils.get_alpha(matting_img)
        mask = utils.get_mask(alpha)
        trimap = utils.get_trimap(alpha)
        image_name = matting.split('/')[-1]
        alpha_path = os.path.join(alpha_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)
        trimap_path = os.path.join(trimap_dir, image_name)
        alpha_path = alpha_path.replace('\\', '/')
        mask_path = mask_path.replace('\\', '/')
        trimap_path = trimap_path.replace('\\', '/')
        cv2.imwrite(alpha_path, alpha)
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(trimap_path, trimap)
        if add_mask_paths:
            new_image_paths.append([clip, matting, alpha_path, mask_path, trimap_path])
        else:
            new_image_paths.append([clip, matting])
    return new_image_paths


if __name__ == '__main__':
    root_dir = '/home/disk1/datas/photo_matting'
    train_txt_path = './train.txt'
    val_txt_path = './val.txt'
    image_paths = get_image_paths(root_dir=root_dir)
    image_paths = save_masks_alphas(image_paths, root_dir)
    train_image_paths, val_image_paths = split_images(image_paths)
    save_image_path(train_image_paths, train_txt_path)
    save_image_path(val_image_paths, val_txt_path)
