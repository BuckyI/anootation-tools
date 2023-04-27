import pycocotools.mask as mask_utils
from pycocotools.mask import encode, decode
import numpy as np
import cv2
import json
import os
from pathlib import Path
import datetime


def file2mask(path: str):
    "load binary mask from image file"
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("mask is None, check your file: {}".format(path))
    return mask > 0


def mask2file(mask: np.ndarray, path: str = "example.jpg", bool2int=True, size=None):
    "save binary mask to image file"
    # 将 bool 转换为 int 图片矩阵
    img_mask = mask.astype(np.uint8) * 255 if bool2int else mask
    if size is not None:
        img_mask = cv2.resize(img_mask, size, cv2.INTER_CUBIC)
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
    encoded_mask["counts"] = str(encoded_mask["counts"], "utf-8")
    return encoded_mask


def bounding_box_from_mask(mask: np.ndarray):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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


def create_ann_file(image_dir, annotation_path="annotation.json", match="*.jpg"):
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
        "contributor": "Zhang & Wang",
        "description": "sick leaves :)",
        "url": None,
        "date_created": formatted_time,
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
    images = list(Path(image_dir).glob(match))
    for image_id, image in enumerate(images):
        # 获取图像信息
        image_path = str(image)
        height, width, channels = cv2.imread(image_path).shape
        # 添加图像信息到 coco 数据集字典中
        coco_data["images"].append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image.name,
                "license": None,
                "url": None,
                "date_captured": None,
            }
        )

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
    json.dump(coco_data, open(annotation_path, "w"))


def merge_annotations(dir):
    """merge annotations from multiple mask files
    NOTE: follow the standard file structure!
    """
    # load annotation file
    ann_file = os.path.join(dir, "annotation.json")
    data = json.load(open(ann_file))
    annotations = data["annotations"]
    assert len(annotations) == 0, f"{ann_file} have annotations, check it"

    # get annotations from mask images
    for image in data["images"]:
        name = image["file_name"]
        mask_path = os.path.join(dir, "masks", name).replace("\\", "/")
        assert os.path.exists(mask_path), f"{name} don't have a mask!"
        mask = file2mask(mask_path)

        annotation = parse_mask_to_coco(
            image_id=image["id"],
            anno_id=len(annotations),
            image_mask=mask,
            category_id=1,
        )
        annotations.append(annotation)
    else:
        # data['annotations'] = annotations
        json.dump(data, open(ann_file, "w"))
