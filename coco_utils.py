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


def mask2file(mask: np.ndarray, path: str = "example.jpg", bool2int=True):
    "save binary mask to image file"
    # 将 bool 转换为 int 图片矩阵
    img_mask = mask.astype(np.uint8) * 255 if bool2int else mask
    # 保证目标文件夹存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        print("mask2file overwrite {}".format(path))
    cv2.imwrite(path, img_mask)


def resize_mask(mask: np.ndarray, size: tuple):
    "input the bool like mask and resize it to the given size"
    assert mask.dtype == np.bool_, "mask must be bool"
    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, size, cv2.INTER_CUBIC)
    return mask > 0


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


class Annotation:
    "used to store annotations of a single image"

    def __init__(self, workdir: str, filename: str):
        self.workdir = Path(workdir)
        self.masksdir = self.workdir / "masks"
        self.filepath = Path(workdir, filename)
        self.name = self.filepath.stem
        assert self.filepath.exists()

        self.annpath = self.masksdir / f"{self.name}.json"
        # standard form {cat1_id: [item1, item2], cat2_id: [item3, item4]}
        self.anns = json.load(open(self.annpath, "r")) if self.annpath.exists() else {}
        self.masks = {}  # {i: [] for i in self.category}

    def files2masks(self):
        for category_id, paths in self.anns.items():
            masks = []
            for path in paths:
                path = self.masksdir / path
                assert path.exists(), "mask {} doesn't exist!".format(path)
                mask = file2mask(str(path))
                masks.append(mask)
            else:
                self.masks[category_id] = masks

    def masks2files(self):
        for category_id, masks in self.masks.items():
            files = []
            for idx, mask in enumerate(masks):
                mask_path = self.masksdir / self.filepath.name
                mask_path = mask_path.with_stem(
                    self.name + " " + str(category_id) + " " + str(idx)
                )
                mask2file(mask, str(mask_path))

                file = str(mask_path.relative_to(self.masksdir)).replace("\\", "/")
                files.append(file)
            else:
                self.anns[category_id] = files
        else:
            json.dump(self.anns, open(self.annpath, "w"))
            print("saved to {}".format(self.annpath))


def init_COCO(image_dir, annotation_path="annotation.json", match="*.jpg"):
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [],
        "annotations": [],
    }

    # info
    now = datetime.datetime.now()
    coco_data["info"] = {
        "year": now.strftime("%Y"),
        "version": None,
        "contributor": "Zhang & Wang",
        "description": "sick leaves :)",
        "url": None,
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
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
            "id": 0,
            "name": "leave",
            "supercategory": None,
        },
        {
            "id": 1,
            "name": "dot",
            "supercategory": None,
        },
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

        anns = Annotation(dir, name)
        anns.files2masks()
        if not anns.masks:
            continue
        for category, masks in anns.masks.items():
            for mask in masks:
                annotation = parse_mask_to_coco(
                    image_id=image["id"],
                    anno_id=len(annotations),
                    image_mask=mask,
                    category_id=category,
                )
                annotations.append(annotation)
    else:
        # data['annotations'] = annotations
        json.dump(data, open(ann_file, "w"))
