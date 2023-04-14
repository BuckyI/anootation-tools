from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys


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


# 导入图片
image = cv2.imread('assets/Snipaste_2023-04-13_13-56-00.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 加载模型
sam_checkpoint = "assets/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

# 获得 mask
logits = None
input_points = []
input_labels = []
masks = None
while True:
    # 展示图片结果
    # fig, ax = plt.subplots()
    plt.imshow(image)
    plt.axis('off')
    # ax.imshow(image)
    if masks is not None:
        show_mask(masks, plt.gca())
        show_points(np.array(input_points), np.array(input_labels), plt.gca())

    # 读取输入的点
    points = plt.ginput(n=2, timeout=-1)
    if not points:
        print("finished")
        break
    input_points.extend(points)
    input_labels.extend([1, 0])

    # get mask
    masks, __, logits = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=np.array(input_labels),
        mask_input=logits,
        multimask_output=False,
    )

# 展示结果
plt.imshow(image)
plt.axis('off')
show_mask(masks, plt.gca())
show_points(np.array(input_points), np.array(input_labels), plt.gca())
plt.show()
