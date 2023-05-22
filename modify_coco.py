import os
import cv2
import json
import datetime
import logging
import shutil
import matplotlib
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from coco_utils import limit_image_size
import logging

matplotlib.use("svg")

"""
load COCO annotation, modify, and export a new dataset
"""


def export_coco_file(prefix, data=[], categories=[], path="annotation.json"):
    """
    data = {basename: annonations}
    basename: basename of image file
    annonations: list of coco annotations
    categories: category name
    """
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
        "description": "polygon version of sick leaves :)",
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

    # categories
    coco_data["categories"] = [
        {"id": i, "name": name, "supercategory": None}
        for i, name in enumerate(categories)
    ]

    for img_id, image in enumerate(data.keys()):
        logging.info("exporting annotation of image {}".format(image))
        # load img
        img_path = prefix + image
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        coco_data["images"].append(
            {
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": image,
                "license": None,
                "url": None,
                "date_captured": None,
            }
        )
        # load annotations
        # segms are the same form of annotation, just need to update infos
        anns = data[image]
        for ann in anns:
            segm = ann["segmentation"]
            rles = mask_utils.frPyObjects(segm, h, w)
            rle = mask_utils.merge(rles)
            area = mask_utils.area(rle)
            bbox = mask_utils.toBbox(rle)

            annotation = {
                "id": len(coco_data["annotations"]),
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": bbox.tolist(),
                "area": float(area),
                "iscrowd": 0,
                "segmentation": ann["segmentation"],
            }
            coco_data["annotations"].append(annotation)

    if os.path.exists(path):
        logging.warning("%s already exists, removed the old one", path)
    json.dump(coco_data, open(path, "w"))
    logging.info("coco annotation file saved to {}".format(path))


def main(img_prefix, work_dir, annFile, test_set=[], size_limit=(1024, 1024)):
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    else:
        shutil.rmtree(work_dir)
        os.mkdir(work_dir)
        logging.warning("old folder removed: {}".format(work_dir))

    # load COCO
    logging.info("loaded coco file")
    coco = COCO(annFile)
    data = dict()

    # load requred imgs (with cat 1 'dots')
    annotated_imgs = coco.getImgIds(catIds=[1])
    annotated_imgs = coco.loadImgs(annotated_imgs)
    for i, img in enumerate(annotated_imgs):
        filename = img["file_name"].strip("data\\")  # custom prefix
        assert os.path.exists(img_prefix + img["file_name"]), f"img {i} doesn't exits!"
        annIds = coco.getAnnIds(imgIds=img["id"], catIds=coco.getCatIds())
        anns = coco.loadAnns(annIds)
        data[filename] = anns
    logging.info("%s imgs with annotation 'dots' are loaded", len(annotated_imgs))

    # process train & test dataset
    train_dir = work_dir + "train/"
    test_dir = work_dir + "test/"
    display_dir = work_dir + "display/"
    for t in [train_dir, test_dir, display_dir]:
        os.mkdir(t)
    train_set = [image for image in data if image not in test_set]
    test_set = [image for image in test_set if image in data]  # omit images not in data
    logging.info("dataset splitted: %s train, %s test", len(train_set), len(test_set))
    for filename, annotations in data.items():
        logging.info("processing: %s ......", filename)
        img_path = img_prefix + "data/" + filename
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)

        if filename in train_set:
            cv2.imwrite(train_dir + filename, img)
        elif filename in test_set:
            # limit size (big picture -> OOM)
            img, scale = limit_image_size(img, size_limit)
            cv2.imwrite(test_dir + filename, img)
            if scale != 1:  # modify sementation size
                for ann in annotations:
                    seg = ann["segmentation"]
                    seg[0] = [scale * x for x in seg[0]]
        # save display annotations
        plt.figure()
        plt.imshow(img)
        coco.showAnns(annotations)
        plt.axis("off")
        plt.savefig(display_dir + filename + ".svg")

    logging.info("exporting train set")
    export_coco_file(
        train_dir,
        {image: data[image] for image in train_set},
        categories=["leave", "dots"],
        path=train_dir + "annotation.json",
    )
    logging.info("exporting test set")
    export_coco_file(
        test_dir,
        {image: data[image] for image in test_set},
        categories=["leave", "dots"],
        path=test_dir + "annotation.json",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_set = [
        "10041_00000004.jpg",
        "10041_00000012.jpg",
        "10041_00000013.jpg",
        "10041_00000014.jpg",
        "10041_00000015.jpg",
        "10041_00000017.jpg",
        "10041_00000018.jpg",
        "10041_00000019.jpg",
        "10041_00000021.jpg",
        "10041_00000022.jpg",
        "10041_00000029.jpg",
        "10041_00000049.jpg",
        "10041_00000084.jpg",
        "10041_00000123.jpg",
        "10041_00000129.jpg",
        "10041_00000132.jpg",
        "10041_00000152.jpg",
        "10041_00000155.jpg",
    ]
    main(
        img_prefix="dataset/polygon/",
        work_dir="dataset/polygon/output/",
        annFile="dataset/polygon/annotation.json",
        test_set=test_set,
    )
