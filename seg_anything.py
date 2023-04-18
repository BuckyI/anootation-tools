import pickle
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2


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


def get_mask(image: np.ndarray):
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
                        np.array(input_labels), plt.gca())
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


if __name__ == '__main__':
    image = load_image('assets/Snipaste_2023-04-13_13-56-00.jpg')
    masks = get_mask(image)

    # 展示结果
    plt.close()
    plt.imshow(image)
    plt.axis('on')
    show_mask(masks, plt.gca())
    plt.show()
