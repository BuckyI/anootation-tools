import pycocotools.mask as mask_utils
from pycocotools.mask import encode, decode
import numpy as np
import cv2
import json
import os
import datetime


def file2mask(path: str):
    "load binary mask from image file"
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("mask is None, check your file: {}".format(path))
    return mask > 0


def mask2file(mask: np.ndarray, path: str = 'example.jpg'):
    "save binary mask to image file"
    img_mask = mask.astype(np.uint8) * 255
    # 保证目标文件夹存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print("mask2file overwrite {}".format(path))
    cv2.imwrite(path, img_mask)


def encode_mask(mask: np.ndarray):
    """
    Encode a binary mask using RLE. (bool array)
    It's ok if the mask is 0 or 255 (8 bit images)
    """
    fortran_binary_mask = np.asfortranarray(mask)
    encoded_mask = encode(fortran_binary_mask)
    encoded_mask['counts'] = str(encoded_mask['counts'], 'utf-8')
    return encoded_mask


def bounding_box_from_mask(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = []
    for contour in contours:
        all_contours.extend(contour)
    convex_hull = cv2.convexHull(np.array(all_contours))
    x, y, w, h = cv2.boundingRect(convex_hull)
    return x, y, w, h


def parse_mask_to_coco(image_id, anno_id, image_mask, category_id):
    x, y, width, height = bounding_box_from_mask(image_mask)
    encoded_mask = encode_mask(image_mask)

    annotation = {
        "id": anno_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(x), float(y), float(width), float(height)],
        "area": float(width * height),
        "iscrowd": 0,
        "segmentation": encoded_mask,
    }
    return annotation


def create_ann_file(image_dir, annotation_path='annotation.json'):
    image_files = os.listdir(image_dir)

    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [],
        "annotations": [],
    }

    # info
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    coco_data["info"] = {
        "year": 2023,
        "version": None,
        "contributor": 'Zhang & Wang',
        "description": 'sick leaves :)',
        "url": None,
        "date_created": formatted_time
    }

    # licenses
    coco_data["licenses"] = [
        {
            "id": 0,
            "name": None,
            "url": None,
        }
    ]

    # images
    for image_id, image_file in enumerate(image_files):
        # 获取图像信息
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        # 添加图像信息到 coco 数据集字典中
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_file,
            "license": None,
            "url": None,
            "date_captured": None
        })

    # categories
    coco_data["categories"] = [
        {
            "id": 1,
            "name": "dot",
            "supercategory": None,
        }
    ]

    if os.path.exists(annotation_path):
        print("already exists, removed the old one")
    json.dump(coco_data, open(annotation_path, 'w'))
