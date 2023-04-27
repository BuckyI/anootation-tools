# Annotation Tools

Anything that helps with annotation! :)

match_annotate(deprecated): 使用预先定义的图片进行屏幕图像匹配，自动点点，方便使用 labelme 进行标注。不过现在有了 Segment Anything，就不需要它了！ 

标注的几个存在形式：
1. binary masks (ndarray) (Segment Anything 获得)
2. image files
3. RLE segmentation
4. annotation (a complete form of annotation)

```mermaid
graph LR
1[binary mask]-->|mask2file|2[image file]-->|file2mask|1
1-->|encode_mask|3[RLE]
3-->4[annotation]
```

假设指定图片文件夹为 "images"，约定标注时的文件结构(standard file structure)：
- 图片都直接位于 "images" 且后缀名为".jpg"
- 标注文件在 `images/annotation.json` 处生成，标注文件内图片的 filename 只包括文件名
- mask images 存放于 `images/mask` 且与对应的图片重名
