import pickle
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import coco_utils
from pycocotools.coco import COCO
import pickle
import os


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def load_image(path: str):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_mask(image: np.ndarray, predictor: SamPredictor = None):
    if predictor is None:
    # 加载模型
    sam_checkpoint = "assets/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)

    predictor.set_image(image)

    # 获得 mask
    input_points = []
    input_labels = []
    logits = None
    mask = None
    while True:
        # 展示图片结果
        plt.imshow(image)
        plt.axis('off')
        if mask is not None:
            show_mask(mask, plt.gca())
            show_points(np.array(input_points),
                        np.array(input_labels), plt.gca(), marker_size=80)
            plt.draw()
        # 读取输入的点
        points = plt.ginput(n=2, timeout=-1)
        if not points:
            print("finished")
            break
        input_points.extend(points)
        input_labels.extend([1, 0])

        # get mask
        mask, __, logits = predictor.predict(
            point_coords=np.array(input_points),
            point_labels=np.array(input_labels),
            mask_input=logits,
            multimask_output=False,
        )
    return mask[0]


class Annotator:
    def __init__(self, image_dir: str, annotation: str):
        # 加载模型
        sam_checkpoint = "assets/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)

        # annotation settings
        self.annotation = COCO(annotation)
        self.image_dir = image_dir

        self.current_img = None

    def mask_path(self, filename: str):
        return os.path.join(self.image_dir, 'masks', os.path.basename(filename)).replace("\\", "/")

    def annotate_image(self, filename: str):
        # def click_save(event):
        #     if event.button == 1:  # 左键点击保存
        #         coco_utils.mask2file(mask, self.mask_path(filename))
        #     elif event.button == 3:  # 右键点击取消
        #         plt.close()
        #         self.annotate_image(filename)

        image = load_image(os.path.join(self.image_dir, filename))
        mask = get_mask(image, predictor=self.predictor)
        # confirm
    plt.close()
    plt.imshow(image)
    plt.axis('on')
        show_mask(mask, plt.gca())
        # plt.connect('button_press_event', click_save)
    plt.show()
        coco_utils.mask2file(mask, self.mask_path(filename))

    def search_unannotated_images(self):
        for filename in os.listdir(self.image_dir):
            if os.path.isdir(os.path.join(self.image_dir, filename)):
                continue
            if os.path.exists(self.mask_path(filename)):
                continue
            else:
                yield filename

    def annotate_images(self):
        for filename in self.search_unannotated_images():
            self.annotate_image(filename)


if __name__ == '__main__':
    s = Annotator("images", "annotation.json").annotate_images()


