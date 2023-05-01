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
from typing import Union, Tuple


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
        # color = np.array([30 / 255, 144 / 255, 255 / 255, 0.4]) # 蓝色
        color = np.array([129 / 255, 20 / 255, 184 / 255, 0.5])  # 紫色
        # color = np.array([239 / 255, 83 / 255, 80 / 255, 0.5])  # 红色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_result(image, mask, title="") -> dict:
    "show the final mask and decide what's next"

    def sett(event):
        "press key to set next action"
        if event.key == "enter":  # 保存并退出
            result["img_ok"] = True
            plt.close(fig)
        elif event.key == "contrl+enter":  # 保存并退出
            result["img_ok"] = False
            plt.close(fig)
        elif event.key == "delete":  # 退出
            result["mask_ok"] = False
            plt.close(fig)

    def on_move(event):
        "add vline and hline to compare two picture"
        if event.inaxes:
            for l in lines["vline"]:
                l.set_xdata(event.xdata)
            for l in lines["hline"]:
                l.set_ydata(event.ydata)
            event.inaxes.figure.canvas.draw_idle()

    result = {"mask_ok": True, "img_ok": True}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24.73, 10.58))
    fig.suptitle(title)
    ax1.axis("off"), ax1.imshow(image)
    ax2.axis("off"), ax2.imshow(image), show_mask(mask, ax2)
    # style
    style = {"color": "black", "alpha": 0.5}
    lines = {
        "vline": [ax.axvline(**style) for ax in (ax1, ax2)],
        "hline": [ax.axhline(**style) for ax in (ax1, ax2)],
    }

    # config control action
    fig.canvas.mpl_connect("key_press_event", sett)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda e: plt.close() if e.button == 2 else None,
    )
    plt.show()
    return result


def load_image(path: Union[str, Path]):
    "load image from path"
    image = cv2.imread(str(path))
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


def get_mask(image: np.ndarray, predictor: SamPredictor = None, hint: str = "annotate"):
    if predictor is None:
        predictor = load_model("vit_h")

    predictor.set_image(image)

    # setup
    input_points = []
    input_labels = []
    logits = None
    mask = None

    fig, ax = plt.subplots(figsize=(25, 14), dpi=100)
    ax.set_title(hint)
    ax.axis("off")

    while True:
        # 展示图片结果
        ax.cla()
        ax.imshow(image)
        if mask is not None:
            show_mask(mask, ax)
            show_points(
                np.array(input_points),
                np.array(input_labels),
                ax,
                marker_size=80,
            )
            fig.canvas.draw_idle()
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

    plt.close(fig)
    # when no input_points are given, the `mask` will be None
    return mask[0] if mask is not None else np.zeros(image.shape[:2], dtype=bool)


def image_chunks(image: np.ndarray, size: int = 100) -> Tuple[slice, np.ndarray]:
    """split the image into small chunks,
    yield the chunk and the slice index"""
    height, width, _ = image.shape
    row_blocks = height // size + (height % size != 0)
    col_blocks = width // size + (width % size != 0)
    for row in range(row_blocks):
        for col in range(col_blocks):
            x0, y0 = col * size, row * size
            x1, y1 = min(x0 + size, width), min(y0 + size, height)
            index = (slice(y0, y1), slice(x0, x1))
            yield index, image[index]


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

        self.current_img_ann = None
        self.current_img_ok = False
        self.current_catid = 0
        self.current_mask_ok = False

    @property
    def current_catid(self):
        return self._category

    @current_catid.setter
    def current_catid(self, v):
        keys = list(self.annotation.cats.keys())
        self._category = keys[v % len(keys)]

    def category_name(self, idx):
        return self.annotation.cats[idx]["name"]

    def status(self):
        name = self.current_img_ann.filename
        image_ok = "finished annotation" if self.current_img_ok else "not done!"
        mask_ok = "use this mask" if self.current_mask_ok else "drop this mask"
        category_name = self.annotation.cats[self.current_catid]["name"]
        return (
            f"Annotating image: {name}\n"
            f"Category: {category_name}\n"
            f"Mask: {mask_ok}; \n"
            f"Image: {image_ok};"
        )

    def annotate_image(self, filename: str):
        def set_action(self, event):
            "press key to set next action"
            if event.key == "left":  # 重新标注当前 mask
                self.current_mask_ok = False
            elif event.key == "right":  # 使用当前 mask
                self.current_mask_ok = True
            elif event.key == "up":  # 更改类别
                self.current_catid += 1
            elif event.key == "down":  # 更改类别
                self.current_catid -= 1
            elif event.key == "enter":  # 继续标注本图片
                self.current_img_ok = False
            plt.title(self.status()), plt.draw()

        # initialize
        ann = self.current_img_ann
        ann.masks = {i: [] for i in self.annotation.cats.keys()}

        # load image
        image = load_image(ann.filepath)
        small_image, _ = resize_image(image, 800, 800)  # 使用小尺寸图片标注

        # annotate
        self.current_img_ok = False
        while not self.current_img_ok:
            # get mask
            self.current_mask_ok = False
            while not self.current_mask_ok:
                mask = get_mask(
                    small_image,
                    predictor=self.predictor,
                    hint=self.status(),
                )

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
                ann.masks[self.current_catid].append(mask)
        else:
            # save annotations
            ann.save_data()

    def annotate_images(self):
        image_ids = self.annotation.getImgIds()
        images = self.annotation.loadImgs(image_ids)
        for image in images:
            filename = image["file_name"]
            ann = coco_utils.Annotation(self.image_dir, filename)
            if ann.files:  # and ann.anns.get("1"):  # 存在病害的标注信息视为已经标注完成
                continue
            else:
                # annotate this image
                self.current_img_ann = ann  # resume annotation
                self.annotate_image(filename)


if __name__ == "__main__":
    workdir = "dataset/images/"
    s = Annotator(
        workdir, model="vit_l", annotation=workdir + "annotation.json"
    ).annotate_images()
    coco_utils.export_COCO(workdir, dst="test_annotation.json")
