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
import matplotlib.pyplot as plt
from typing import Union
import threading
from typing import Union, Tuple, List


def run_in_thread(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


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
    """input the bool like mask and resize it to the given size
    size = (w, h)
    """
    assert mask.dtype == np.bool_, "mask must be bool"
    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, size, cv2.INTER_CUBIC)
    return mask > 0


def limit_image_size(image: np.ndarray, size: tuple) -> Tuple[np.ndarray, float]:
    """
    将图片大小缩放到不超过指定的最大高度和宽度，并返回缩放比例。
    size = (max_height, max_width)
    """
    # height, width, _ = image.shape
    # if height > max_height or width > max_width:
    #     scale = min(max_height / height, max_width / width)
    #     new_height = int(height * scale)
    #     new_width = int(width * scale)
    #     resized_image = cv2.resize(image, (new_width, new_height))
    # else:
    #     scale = 1.0
    #     resized_image = image.copy()

    # return resized_image, scale

    # 简易版代码
    old_shape = np.array(image.shape[:2])
    ratio = np.array(size) / np.array(image.shape[:2])
    scale = min(ratio.clip(min=0, max=1))
    shape = (old_shape * scale).astype(int)  # target shape
    resized_image = cv2.resize(image, tuple(reversed(shape)))  # cv2 uses (w, h)
    return resized_image, scale


def morph_close(mask: np.ndarray, size=5, iterations=5):
    "闭运算 去除 mask 的细小空洞 (False dots)"
    assert mask.dtype == np.bool_, "mask must be bool"
    kernel = np.ones((size, size), np.uint8)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
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

    def __init__(
        self, workdir: Union[str, Path], filename: str, *, category: dict = None
    ):
        # init
        self.workdir = Path(workdir)
        self.masksdir = self.workdir / "masks"  # store files
        self.masksdir.mkdir(exist_ok=True)  # make sure folder exists
        self.filepath = Path(workdir, filename)  # image to be annotated
        assert self.filepath.exists()
        self.filename = filename  # str
        self.image = None  # ndarray
        self.finished = False  # bool
        self.cat = category if category else ["leave", "dot"]  # list of str
        self.masks = []  # list of (ndarray, category_id)
        self.visualize = None  # ndarray (h, w, 3)

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
            "visualize": self.visualize_masks(),
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
            self.visualize = data["visualize"]
            return data
        else:
            img = cv2.imread(str(self.filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image = img
            self.visualize = img
            return None

    @run_in_thread
    def save_data(self):
        if self.image is None:
            img = cv2.imread(str(self.filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image = img
        # drop all False masks
        self.masks = [(mask, catid) for mask, catid in self.masks if np.any(mask)]
        # delet previous mask image
        for i in Path(self.masksdir).glob(self.filename.strip(".jpg") + "*.jpg"):
            i.unlink()
        if self.annpath.exists():
            print("overwrite {}".format(self.annpath))
        data = self.data
        with open(self.annpath, "wb") as f:
            pickle.dump(data, f)
            logging.info("save data of {} to {}".format(self, self.annpath))

    def save_masks(self):
        "visualize masks into image file"
        for idx, (mask, catid) in enumerate(self.masks):
            path = self.masksdir / "{} #id {} #cat {}.jpg".format(
                self.filename.strip(".jpg"),
                str(idx),
                str(catid),
            )
            mask2file(mask, str(path))
            logging.info("save mask image to {}".format(path))

    def visualize_masks(self) -> np.ndarray:
        "使用 cv2 简单展示标注，如果使用 matplotlib 速度会很慢，而且会有问题"
        img = self.image.copy()
        cover = np.zeros(self.image.shape, dtype=img.dtype)  # all masks with color
        cover_mask = np.zeros(self.image.shape[:2], dtype=bool)  # all masks together
        for mask, _ in sorted(self.masks, key=lambda x: x[1]):
            cover_mask = cover_mask | mask
            color = np.random.randint(0, 256, size=(3,)).reshape(1, 1, -1)
            mask_image = (mask[..., None] * color).astype(img.dtype)
            # get the overlapping area 覆盖重叠区域旧的内容
            overlap = cv2.bitwise_and(cover, cover, mask=mask.astype(np.int8))
            cover = cover - overlap + mask_image
        else:
            # 图片和标注合并在一起，保留标注下方的图片
            overlap = cv2.bitwise_and(img, img, mask=cover_mask.astype(np.int8))
            img = img - overlap + cv2.addWeighted(overlap, 0.4, cover, 0.6, 0)

        self.visualize = img
        path = self.masksdir / "{} #FINAL.jpg".format(self.filename.strip(".jpg"))
        # opencv 使用 BGR 处理图片，因此保存时转换一下，否则颜色不对
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), img)
        logging.info("save visualize of {} to {}".format(self, path))
        return self.visualize

    def add_mask(self, mask, category_id):
        assert category_id in range(len(self.cat)), "category_id not in cat"
        assert mask.shape == self.image.shape[:2], "mask shape mismatch"
        self.masks.append((mask, category_id))

    @property
    def splitted_masks(self):
        masks = {i: [] for i in range(len(self.cat))}
        for mask, cat in self.masks:
            masks[cat].append(mask)
        return masks

    def __str__(self):
        result = f"Annotation of {self.filename} with {len(self.masks)} masks. "
        for catid, masks in self.splitted_masks.items():
            result += f"{len(masks)} {self.cat[catid]}s"
        return result


def export_coco_file(
    workdir: Union[str, Path],
    images: list,
    categories: list,
    filename="annotation.json",
    size_limit=None,
):
    """Create COCO annotations from multiple pickle files
    usage:
    - follow the standard file structure!
    - images, categories are from Annotator
    - size_limit: sometimes, the image shape could be too big to be utilized, limit it!
    """
    workdir = Path(workdir)
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
    for img in images:
        coco_data["images"].append(
            {
                "id": img["id"],
                "width": img["width"],
                "height": img["height"],
                "file_name": img["file_name"],
                "license": None,
                "url": None,
                "date_captured": None,
            }
        )

    # categories
    coco_data["categories"] = [
        {"id": i, "name": name, "supercategory": None}
        for i, name in enumerate(categories)
    ]

    # get annotations from mask images
    for image in coco_data["images"]:
        name = image["file_name"]
        anns = Annotation(workdir, name)
        if not anns.masks:
            logging.error("no mask in {}".format(name))
            continue

        # if limit size, then preprocess annotation
        if size_limit is None:
            masks = anns.masks
        else:
            # get resized img
            img, scale = limit_image_size(anns.image, size_limit)
            h, w, _ = img.shape
            image["height"], image["width"] = h, w  # update coco data
            # save resized img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            export_dir = workdir / "limited"
            export_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(export_dir / name), img)
            # get resized masks
            masks = [
                (
                    resize_mask(mask, (w, h)) if scale != 1 else mask,
                    cat,
                )
                for mask, cat in anns.masks
            ]

        for mask, catid in masks:
            annotation = parse_mask_to_coco(
                image_id=image["id"],
                anno_id=len(coco_data["annotations"]),
                image_mask=mask,
                category_id=catid,
            )
            coco_data["annotations"].append(annotation)

    path = workdir / filename
    if path.exists():
        logging.warning("%s already exists, removed the old one", path)
    json.dump(coco_data, open(path, "w"))
    logging.info("coco annotation file saved to {}".format(path))
