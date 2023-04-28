import pickle
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import coco_utils
from pycocotools.coco import COCO
import pickle
import os
import requests
from tqdm import tqdm
from urllib.parse import urlsplit
from pathlib import Path
import json


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def load_image(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def download(url, dir=""):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("Content-Length", 0))
    filepath = os.path.join(dir, urlsplit(url).path.split("/")[-1])
    with open(filepath, "wb") as f, tqdm(
        total=total_size, unit="iB", unit_scale=True
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    print("Download completed! File saved as {}".format(filepath))
    return filepath


def load_model(model="vit_h"):
    "model: vit_h, vit_l, vit_b"
    models = {
        "vit_h": (
            "assets/sam_vit_h_4b8939.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        ),
        "vit_l": (
            "assets/sam_vit_l_0b3195.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        ),
        "vit_b": (
            "assets/sam_vit_b_01ec64.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        ),
    }
    if model not in models:
        raise ValueError("model must be one of {}".format(models.keys()))
    sam_checkpoint, sam_url = models[model]
    if not os.path.exists(sam_checkpoint):
        download(sam_url, "assets/")

    sam = sam_model_registry[model](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    return SamPredictor(sam)


def get_mask(image: np.ndarray, predictor: SamPredictor = None):
    if predictor is None:
        predictor = load_model("vit_h")

    predictor.set_image(image)

    # 获得 mask
    input_points = []
    input_labels = []
    logits = None
    mask = None
    while True:
        # 展示图片结果
        plt.close()
        plt.figure(figsize=(25.60, 14.40), dpi=100)
        plt.title("annotate")
        plt.imshow(image)
        plt.axis("off")
        if mask is not None:
            show_mask(mask, plt.gca())
            show_points(
                np.array(input_points),
                np.array(input_labels),
                plt.gca(),
                marker_size=80,
            )
            plt.draw()
        # 读取输入的点
        # usage: input point(s), the last one is negetive
        points = plt.ginput(n=-1, timeout=-1)
        if not points:
            print("finished")
            break
        input_points.extend(points)
        input_labels.extend([1] * (len(points) - 1) + [0])

        # get mask
        mask, __, logits = predictor.predict(
            point_coords=np.array(input_points),
            point_labels=np.array(input_labels),
            mask_input=logits,
            multimask_output=False,
        )
    assert mask is not None
    return mask[0]


def resize_image(image, max_height, max_width):
    """
    将图片大小缩放到不超过指定的最大高度和宽度，并返回缩放比例。
    """
    height, width = image.shape[:2]
    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
    else:
        scale = 1.0
        resized_image = image.copy()

    return resized_image, scale


class Annotator:
    def __init__(self, image_dir: str, annotation: str = None, model: str = "vit_h"):
        # 加载模型
        self.predictor = load_model(model)

        # annotation settings
        if annotation is None:
            annotation = os.path.join(image_dir, "annotation.json")
            coco_utils.init_COCO(image_dir, annotation)
        elif not os.path.exists(annotation):
            coco_utils.init_COCO(image_dir, annotation)
        self.annotation = COCO(annotation)
        self.image_dir = image_dir

        self.current_img = None
        self.current_img_ok = False
        self.current_category = 0
        self.current_mask_ok = False
        self.current_annotations = None

    @property
    def current_category(self):
        return self._category

    @current_category.setter
    def current_category(self, v):
        keys = list(self.annotation.cats.keys())
        self._category = keys[v % len(keys)]

    def category_name(self, idx):
        return self.annotation.cats[idx]["name"]

    def mask_path(self, filename, category_name, idx):
        assert self.current_img is not None
        mask_dir = Path(self.image_dir, "masks")
        mask_path = mask_dir / Path(filename)
        mask_path = mask_path.with_stem(
            mask_path.stem + " " + category_name + " " + str(idx)
        )
        assert not mask_path.exists()
        return str(mask_path.relative_to(self.image_dir)).replace("\\", "/")

    def status(self):
        image_ok = "finished" if self.current_img_ok else "continue"
        mask_ok = "use this mask" if self.current_mask_ok else "drop this mask"
        category_name = self.category_name(self.current_category)
        return f"Annotating {category_name} in {self.current_img}\n\
            mask: {mask_ok}; \nimage: {image_ok};"

    def annotate_image(self, filename: str):
        def set_action(self, event):
            "press key to set next action"
            if event.key == "left":  # 重新标注当前 mask
                self.current_mask_ok = False
            elif event.key == "right":  # 使用当前 mask
                self.current_mask_ok = True
            elif event.key == "up":  # 更改类别
                self.current_category += 1
            elif event.key == "down":  # 更改类别
                self.current_category -= 1
            elif event.key == "enter":  # 继续标注本图片
                self.current_img_ok = False
            plt.title(self.status()), plt.draw()

        # load image
        self.current_img = filename
        image = load_image(os.path.join(self.image_dir, filename))
        small_image, _ = resize_image(image, 800, 800)  # 使用小尺寸图片标注

        # initialize
        self.current_annotations = {i: [] for i in self.annotation.cats.keys()}
        self.current_img_ok = False
        while not self.current_img_ok:
            # get mask
            self.current_mask_ok = False
            while not self.current_mask_ok:
                mask = get_mask(small_image, predictor=self.predictor)

                # by default, let this mask be the last mask of this image
                self.current_mask_ok = True
                self.current_img_ok = True
                # confirm if mask is ok
                plt.close()
                plt.figure(figsize=(25.60, 14.40), dpi=100)
                plt.title(self.status())
                plt.imshow(small_image)
                plt.axis("on")
                show_mask(mask, plt.gca())
                plt.connect(
                    "button_press_event",
                    lambda e: plt.close() if e.button == 2 else None,
                )
                plt.connect("key_press_event", lambda e: set_action(self, e))
                plt.show()
            else:  # finally save the good mask
                # resize the mask to the original image size
                size = (image.shape[1], image.shape[0])
                mask = coco_utils.resize_mask(mask, size)
                self.current_annotations[self.current_category].append(mask)
        else:
            # save annotations
            for cat, masks in self.current_annotations.items():
                paths = []
                for j, mask in enumerate(masks):
                    mask_path = self.mask_path(filename, self.category_name(cat), j)
                    path = os.path.join(self.image_dir, mask_path)
                    coco_utils.mask2file(mask=mask, path=path)
                    paths.append(mask_path)
                else:
                    # record the mask files path
                    self.current_annotations[cat] = paths

            ann_file = Path(self.image_dir, "masks", f"{filename}.json")
            assert not ann_file.exists()
            with open(ann_file, "w") as f:
                json.dump(self.current_annotations, f)
                print("saved to {}".format(ann_file))

    def search_unannotated_images(self):
        image_ids = self.annotation.getImgIds()
        images = self.annotation.loadImgs(image_ids)
        for image in images:
            filename = image["file_name"]
            path = Path(self.image_dir, filename)
            assert path.exists(), "{} does not exist".format(path)
            ann_path = Path(self.image_dir, "masks", f"{filename}.json")
            if not ann_path.exists():  # 如果存在同名的 json，认为已经标注完成
                yield filename

    def annotate_images(self):
        for filename in self.search_unannotated_images():
            self.annotate_image(filename)


if __name__ == "__main__":
    workdir = "dataset/images/"
    s = Annotator(
        workdir, model="vit_h", annotation=workdir + "annotation.json"
    ).annotate_images()
    coco_utils.merge_annotations(workdir)
