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
        # usage: input point(s), the last one is negetive
        points = plt.ginput(n=-1, timeout=30)
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
    def __init__(self, image_dir: str, annotation: str = None):
        # 加载模型
        sam_checkpoint = "assets/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)

        # annotation settings
        if annotation is None:
            annotation = os.path.join(image_dir, "annotation.json")
            coco_utils.create_ann_file(image_dir, annotation)
        self.annotation = COCO(annotation)
        self.image_dir = image_dir

        self.current_img = None

    def mask_path(self, filename: str):
        return os.path.join(self.image_dir, 'masks', os.path.basename(filename)).replace("\\", "/")

    def annotate_image(self, filename: str):
        def click_save(self, event):
            "event.button: 1 left click; 3 right click"
            if event.button == 3:  # 右键点击取消
                print("drop this mask")
                self.current_mask_ok = False

        # load image
        image = load_image(os.path.join(self.image_dir, filename))
        small_image, _ = resize_image(image, 800, 800)  # 使用小尺寸图片标注

        # get mask
        self.current_mask_ok = False
        while(not self.current_mask_ok):
            mask = get_mask(small_image, predictor=self.predictor)
            # confirm if mask is ok
            self.current_mask_ok = True
            plt.close()
            plt.imshow(small_image)
            plt.axis('on')
            show_mask(mask, plt.gca())
            plt.connect('button_press_event', lambda e: click_save(self, e))
            plt.show()
        else:  # finally save the good mask
            # resize the mask to the original image size
            size = (image.shape[1], image.shape[0])
            coco_utils.mask2file(mask, self.mask_path(
                filename), size=size)
            print("saved to {}".format(self.mask_path(filename)))

    def search_unannotated_images(self):
        image_ids = self.annotation.getImgIds()
        images = self.annotation.loadImgs(image_ids)
        for image in images:
            path = os.path.join(self.image_dir, image['file_name'])
            assert os.path.exists(path), "{} does not exist".format(path)
            if os.path.exists(self.mask_path(path)):
                # already have a mask
                continue
            else:
                yield image['file_name']

    def annotate_images(self):
        for filename in self.search_unannotated_images():
            self.annotate_image(filename)


if __name__ == '__main__':
    workdir = 'demo_images'
    s = Annotator(workdir).annotate_images()
    coco_utils.merge_annotations(workdir)
