import pickle
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import coco_utils
from coco_utils import limit_image_size
from pycocotools.coco import COCO
import pickle
import os
import requests
from tqdm import tqdm
from urllib.parse import urlsplit
from pathlib import Path
import json
from typing import Union, Tuple, List
import logging


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


def show_result(image, masks: List[np.ndarray], title="") -> dict:
    "show the final mask and decide what's next"

    def record(event):
        """record the last keyboard event, possible keys:
        'left', 'right', 'up', 'down', 'delete', 'enter', 'ctrl+enter', 'control'
        """
        if event.key in ("delete", "enter", "ctrl+enter"):
            key_event[0] = event.key
            plt.close(fig)

    def on_move(event):
        "add vline and hline to compare two picture"
        if event.inaxes:
            for l in lines["vline"]:
                l.set_xdata(event.xdata)
            for l in lines["hline"]:
                l.set_ydata(event.ydata)
            event.inaxes.figure.canvas.draw_idle()

    # resize image and masks to accelerate plotting
    max_h = max_w = 750
    image, _ = limit_image_size(image, (max_h, max_w))
    for i, mask in enumerate(masks):
        masks[i] = coco_utils.resize_mask(mask, tuple(reversed(image.shape[:2])))

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24.73, 10.58))
    fig.suptitle(title)
    for ax in (ax1, ax2):
        ax.axis("off"), ax.imshow(image)
    for mask in masks:
        show_mask(mask, ax2, random_color=True)
    # style
    style = {"color": "black", "alpha": 0.5}
    lines = {
        "vline": [ax.axvline(**style) for ax in (ax1, ax2)],
        "hline": [ax.axhline(**style) for ax in (ax1, ax2)],
    }

    # config control action
    key_event = [None]
    fig.canvas.mpl_connect("key_press_event", record)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda e: plt.close() if e.button == 2 else None,
    )
    plt.show()
    return key_event[0]


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


def get_mask(
    image: np.ndarray, predictor: SamPredictor = None, hint: str = "annotate"
) -> Tuple[np.ndarray, str]:
    def record(event):
        """record the last keyboard event, possible keys:
        'left', 'right', 'up', 'down', 'delete', 'enter', 'ctrl+enter', 'control'
        """
        key_event[0] = event.key

    if predictor is None:
        predictor = load_model("vit_h")

    predictor.set_image(image)

    # setup
    input_points = []
    input_labels = []
    logits = None
    mask = None

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(25, 14), dpi=100)
    fig.suptitle(hint)
    for ax in (ax1, ax2):
        ax.axis("off"), ax.imshow(image)

    # record key
    key_event = [None]  # record keyboard event
    cid = fig.canvas.mpl_connect("key_press_event", record)

    while True:
        # 展示图片结果
        ax2.cla()
        ax2.axis("off"), ax2.imshow(image)
        if mask is not None:
            show_mask(mask, ax2)
            show_points(
                np.array(input_points),
                np.array(input_labels),
                ax2,
                marker_size=80,
            )
            fig.canvas.draw_idle()
        # 读取输入的点
        # usage: input point(s), the last one is negetive

        points = plt.ginput(n=-1, timeout=-1)
        if not points:  # 没有任何输入时，结束
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

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)
    # when no input_points are given, the `mask` will be None
    mask = mask[0] if mask is not None else np.zeros(image.shape[:2], dtype=bool)
    return mask, key_event[0]


def image_chunks(image: np.ndarray, size: int = 100) -> Tuple[slice, np.ndarray]:
    """split the image into small chunks,
    yield the chunk and the slice index"""
    assert type(size) == int
    height, width, _ = image.shape
    row_blocks = height // size + (height % size != 0)
    col_blocks = width // size + (width % size != 0)
    for row in range(row_blocks):
        for col in range(col_blocks):
            x0, y0 = col * size, row * size
            x1, y1 = min(x0 + size, width), min(y0 + size, height)
            index = (slice(y0, y1), slice(x0, x1))
            yield index, image[index]


class Annotator:
    def __init__(self, image_dir: str, annotation: str = None, *, model: str = "vit_h"):
        "annotation: data.pickle"
        self.modeltype = model
        self._predictor = None  # lazy load predictor
        self.image_dir = Path(image_dir)
        # load data
        if annotation is None:
            logging.warning(f"initalizing data... Hello!")
            self.init_data(save=True)
        elif not Path(annotation).exists:
            logging.warning(f"{annotation} does not exist, created a new one")
            self.init_data(save=True)
        else:
            assert annotation.endswith(".pickle"), "annotation must be a pickle file"
            self.data = pickle.load(open(annotation, "rb"))

    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = load_model(self.modeltype)
        return self._predictor

    def init_data(self, match="*.jpg", save=True):
        self.data = {"images": [], "categories": ["leave", "dot"]}
        for image_id, image in enumerate(self.image_dir.glob(match)):
            image_path = str(image)
            height, width, channels = cv2.imread(image_path).shape
            # 添加图像信息到 coco 数据集字典中
            self.data["images"].append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": image.name,
                }
            )
        if save:
            path = self.image_dir / "data.pickle"
            if path.exists():
                logging.warning(f"{path} already exists, removed the old one")
            pickle.dump(self.data, open(path, "wb"))
            logging.info("saved data to %s" % path)

    def annotate_images(self, visualize=True):
        num_images = len(self.data["images"])
        current_idx = 0
        flag = True
        while flag:
            filename = self.data["images"][current_idx]["file_name"]
            ann = coco_utils.Annotation(
                self.image_dir, filename, category=self.data["categories"]
            )
            logging.info("display %s" % filename)
            key = self.display(
                ann,
                title=(
                    f"{ann}\n"
                    "If annotate dots in this, press 'enter'\n"
                    "If annotate leave in this, press 'space'\n"
                    "'left' ← and 'right' → to change image :)\n"
                    "'ctrl+enter' to label this image as finished\n"
                    "'delete' to clear all masks\n"
                    "'escape' to quit"
                ),
                save2file=visualize,
            )

            if key == "enter":
                self.annotate_dots(ann)
                current_idx += 1
            elif key == " ":
                self.annotate_leaves(ann)
                current_idx += 1
            elif key == "left":
                current_idx -= 1
            elif key == "right":
                current_idx += 1
            elif key == "ctrl+enter":
                ann.finished = True
                current_idx += 1
            elif key == "delete":
                ann.masks = []
                ann.save_data()
            elif key == "escape":
                flag = False

            current_idx % num_images

        logging.info("Goodbye! :)")

    def display(self, current: coco_utils.Annotation, title="", save2file=False):
        def record(event):
            """record the last keyboard event, possible keys:
            "enter", "left", "right", "escape", "ctrl+enter"
            """
            if event.key in [
                "enter",
                " ",
                "left",
                "right",
                "ctrl+enter",
                "delete",
                "escape",
            ]:
                key_event[0] = event.key
                plt.close(fig)

        fig, ax = plt.subplots()
        fig.suptitle(title), ax.axis("off")
        ax.imshow(current.visualize)

        key_event = [None]
        fig.canvas.mpl_connect("key_press_event", record)
        fig.canvas.mpl_connect(
            "button_press_event",
            lambda e: plt.close() if e.button == 2 else None,
        )
        plt.show()
        return key_event[0]

    def annotate_leaves(self, current: coco_utils.Annotation):
        logging.info("start annotating leave in %s", current.filename)
        # initialize
        leave_id = 0
        # 使用小尺寸图片标注，记录原尺寸
        h, w, _ = current.image.shape
        image, _ = limit_image_size(current.image, (800, 800))

        masks = []
        while True:
            mask, key = get_mask(
                image,
                predictor=self.predictor,
                hint="Annotate Leaves in %s" % current.filename,
            )
            logging.info("this mask get: %s", key)
            # press: delete -> reject this mask, reannotate the image
            # press: 'ctrl+enter' -> accept this mask, but continue annotate the image
            # other: accept this mask and move one to the next image
            if key != "delete" and np.any(mask):
                masks.append(mask)
                logging.info("%s get a new leave mask", current.filename)
            if key not in ["delete", "ctrl+enter"]:
                break

        logging.warning("%s gets %s leave masks finally", current.filename, len(masks))
        if len(masks) == 0:  # no mask
            return

        key = show_result(
            image,
            masks,
            (
                f"This is the {len(masks)} leave of this image\n"
                "press any key to continue\n"
                "press 'delete' to reject, and move on :(\n"
                "press 'ctrl+enter' to reannotate this image :|"
            ),
        )
        # press: delete -> reject all masks
        # press: 'ctrl+enter' -> reject all masks, reannotate
        if key == "delete":
            logging.warning("no mask of %s was accepted, return", current.filename)
        elif key == "ctrl+enter":
            logging.warning("no mask of %s was accepted, reannotate", current.filename)
            self.annotate_leaves(current)
        else:  # save masks
            logging.info(
                "%s update masks: %s + %s",
                current.filename,
                len(current.masks),
                len(masks),
            )
            for mask in masks:
                # resize the mask to the original image size
                mask = coco_utils.resize_mask(mask, (w, h))
                current.masks.append((mask, leave_id))
            current.save_data()

    def annotate_dots(self, current: coco_utils.Annotation):
        logging.info("start annotating dots in %s", current.filename)
        # initialize
        leave_id, dot_id = 0, 1
        # find all leaves, background covered
        leaves = [
            current.image * mask[:, :, np.newaxis]
            for mask in current.splitted_masks[leave_id]
        ]
        logging.info("%s has %s leaves to annotate", current.filename, len(leaves))
        # annotate all leaves
        for leave in leaves:
            # Get chunk size:
            # 1. get the span of area containing leave
            # 2. get the longest size `s` (width or height)
            # 2. let `s/3` be the chunk size
            indices = np.where(leave[:, :, 0] > 0)
            h, w = np.max(indices, axis=1) - np.min(indices, axis=1) + 1
            size = int(np.ceil(max(w / 3, h / 3)))

            # split the leave into chunks
            dot_mask = np.zeros(leave.shape[:2], dtype=bool)
            for idx, chunk in image_chunks(leave, size):
                if not np.any(chunk):
                    # some chunks don't have leave part, skip them
                    continue
                while True:
                    mask, key = get_mask(
                        chunk,
                        predictor=self.predictor,
                        hint="Annotate Dots in %s" % current.filename,
                    )
                    logging.info("this mask get: %s", key)
                    # press: delete -> reject this mask, reannotate the chunk
                    # other: accept this mask and move one to the next chunk
                    if key != "delete" and np.any(mask):
                        dot_mask[idx] = mask
                    if key != "delete":
                        break
            logging.info("get a dots mask of %s", current.filename)
            key = show_result(
                leave,
                [dot_mask],
                (
                    f"This is the final mask of image {current.filename}\n"
                    "press any key to continue\n"
                    "press 'delete' to reject, and move on :(\n"
                ),
            )
            # press: delete -> reject the mask
            if key == "delete":
                logging.warning("reject this mask of %s", current.filename)
            else:  # save masks
                # save masks
                logging.info(
                    "%s update masks: %s + 1", current.filename, len(current.masks)
                )
                current.masks.append((dot_mask, dot_id))
                current.save_data()

    def export(self):
        coco_utils.export_coco_file(
            self.image_dir,
            images=self.data["images"],
            categories=self.data["categories"],
        )
        logging.info("export to %s", self.image_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    workdir = "dataset/images/"
    s = Annotator(workdir, model="vit_l", annotation=workdir + "data.pickle")
    # s.annotate_images()
    s.export()
