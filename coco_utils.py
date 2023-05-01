import pycocotools.mask as mask_utils
from pycocotools.mask import encode, decode
import numpy as np
import cv2
import json
import os
from pathlib import Path
import datetime
import pickle
import logging


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
    "used to temporarily store annotations of a single image"

    def __init__(self, dir: str, filename: str, *, category: dict = None):
        # init
        self.workdir = Path(dir)
        self.masksdir = self.workdir / "masks"  # store files
        self.filepath = Path(dir, filename)  # image to be annotated
        assert self.filepath.exists()
        self.filename = filename  # str
        self.image = None  # ndarray
        self.finished = False  # bool
        self.cat = category if category else ["leave", "dot"]  # list of str
        self.masks = []  # list of (ndarray, category_id)

        # load previous annotation
        self.name = self.filepath.stem
        self.annpath = self.masksdir / "{}.pickle".format(filename.strip(".jpg"))
        self.load_data()

    @property
    def data(self):
        return {
            "filename": self.filename,
            "image": self.image,
            "finished": self.finished,
            "category": self.cat,
            "masks": self.masks,
        }

    def load_data(self):
        if self.annpath.exists():
            print("load from {}".format(self.annpath))
            data = pickle.load(open(self.annpath, "rb"))
            assert data["filename"] == self.filename, "annotation file mismatch"
            self.image = data["image"]
            self.finished = data["finished"]
            self.cat = data["category"]
            self.masks = data["masks"]
            return data
        else:
            return None

    def save_data(self):
        if self.image is None:
            img = cv2.imread(str(self.filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image = img
        if self.annpath.exists():
            print("overwrite {}".format(self.annpath))
        with open(self.annpath, "wb") as f:
            pickle.dump(self.data, f)

    def visualize_masks(self):
        "visualize masks into image file"
        for idx, (mask, catid) in enumerate(self.masks):
            path = self.masksdir / "{} #id {} #cat {}.jpg".format(
                self.filename.strip(".jpg"),
                str(idx),
                str(catid),
            )
            mask2file(mask, str(path))

    def add_mask(self, mask, category_id):
        assert category_id in self.cat, "category_id not in cat"
        self.masks.append((mask, category_id))


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


def export_COCO(dir, scr="annotation.json", dst="annotation.json"):
    """merge annotations from multiple pickle files
    NOTE: follow the standard file structure!
    """
    # load annotation file
    scr_file = os.path.join(dir, scr)
    data = json.load(open(scr_file, "r"))
    annotations = data["annotations"]
    assert len(annotations) == 0, f"{scr_file} have annotations, check it"

    # get annotations from mask images
    for image in data["images"]:
        name = image["file_name"]

        anns = Annotation(dir, name)
        if not anns.masks:
            logging.error("no mask in {}".format(name))
            continue
        for mask, catid in anns.masks:
            annotation = parse_mask_to_coco(
                image_id=image["id"],
                anno_id=len(annotations),
                image_mask=mask,
                category_id=catid,
            )
            annotations.append(annotation)
    else:
        # data['annotations'] = annotations
        dst_file = os.path.join(dir, dst)
        json.dump(data, open(dst_file, "w"))
