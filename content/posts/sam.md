+++
author = "Mohamed ABDUL GAFOOR"
date = "2023-04-18"
title = "Revolutionizing Segmentation: Introducing the Segment Anything Model (SAM)"
slug = "sam"
tags = [
    "Segmentation",
    "Machine Learning",
    "Deep Learning"
]
categories = [
    "Artificial Intelligence"
]

+++

In this post, we will be discussing the MetaAI release of the Segment Anything Model (SAM). This is a highly potent framework with the ability to be implemented across a broad range of scientific and technological domains. For those interested in learning more about SAM, please visit the website [SAM.](https://segment-anything.com/)

Computer vision has relied heavily on segmentation, the process of identifying which image pixels belong to an object. However, creating an accurate segmentation model for specific tasks usually requires specialized technical experts and large volumes of carefully annotated in-domain data. The Segment Anything Model aims to democratize segmentation and reduce the need for task-specific modeling expertise, training, and custom data annotation.

SAM has learned a general notion of what objects are, and it can generate masks for any object in any image or video, even including objects and image types that it had not encountered during training. SAM is general enough to cover a broad set of use cases and can be used out of the box on new image "domains" without requiring additional training. This capability is often referred to as zero-shot transfer. To create a dataset to train SAM, the Segment Anything project has developed the Segment Anything 1-Billion mask dataset (SA-1B), the largest-ever segmentation dataset. SA-1B is a highly diverse dataset, covering a wide range of object types, image types, and contexts. 


**SAM overview**

The Segment Anything Model (SAM) is a framework for image segmentation that utilizes an image encoder to output an image embedding. This image embedding is then used by a lightweight encoder to efficiently generate object masks based on various input prompts. The input prompts can include foreground/background points, rough boxes or masks, freeform text, or any information indicating what to segment in an image.

SAM is capable of handling ambiguous prompts, which correspond to more than one object. In such cases, the model can output multiple valid masks, each associated with a confidence score. The image encoder and decoder are designed to operate in real-time, allowing for quick and efficient segmentation of images. This makes SAM a highly potent tool for a broad range of scientific and technological applications, as it simplifies the process of segmenting images and requires minimal technical expertise. It has three main components, image encoder, prompt encoder and mask decoder [paper.](https://scontent-cdg4-2.xx.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=F08W4cSoXYUAX-fJcX-&_nc_ht=scontent-cdg4-2.xx&oh=00_AfBfiRtGEmYW_vHzx4izg2R8vGsGEVLSgf_JRO3D6imSGA&oe=6442D8A7)

{{< figure class="center" src="/images/sam.png" >}}

To evaluate the performance of SAM, we will utilize Google Colab. Install the following libraries using pip.

```Python
%%capture
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install opencv-python pycocotools matplotlib onnxruntime onnx
!pip install jupyter_bbox_widget
```

We use the following versions..
```Python
PyTorch version: 2.0.0+cu118
Torchvision version: 0.15.1+cu118
CUDA is available: True
```

Import the following libraries;

```Python
import cv2
import numpy as np
import PIL
import io
import html
import time
import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from google.colab.patches import cv2_imshow
from google.colab import output
from jupyter_bbox_widget import BBoxWidget
import base64
```

We will define the following functions. These functions were extracted from SAM github [page.](https://github.com/facebookresearch/segment-anything)

```Python
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   
```

Let us load few images and draw a boundingbox;
```Python
imPath = './ele.png'
img = cv2.imread(imPath)
cv2_imshow(img)

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded
    
if using_colab:
    output.enable_custom_widget_manager()

widget = BBoxWidget()
widget.image = encode_image(imPath)
widget
```

Now we must provide path to the weight and model type. There are 3 weights and types ([here](https://github.com/facebookresearch/segment-anything));

* default or vit_h: ViT-H SAM model.
* vit_l: ViT-L SAM model.
* vit_b: ViT-B SAM model.


```Python
coordinates = widget.bboxes
x, y, w, h = coordinates[0]['x'], coordinates[0]['y'], coordinates[0]['width'], coordinates[0]['height']
input_box = np.array([x, y, x + w, y + h])
input_label = np.array([0])

def sam_BB(image, input_box):
  sam_checkpoint = "./sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=DEVICE)

  predictor = SamPredictor(sam)

  predictor.set_image(image)

  masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,)

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  plt.figure(figsize=(10, 10))
  plt.imshow(image)
  show_mask(masks[0], plt.gca())
  show_box(input_box, plt.gca())
  plt.axis('off')
  plt.show()

sam_BB(img, input_box)

```

Following are the segmented images within the boundingbox;
{{< figure class="center" src="/images/seg1.png" >}}


Now we will test the performance of automatically generating object masks with SAM.. For this purpose few biological samples were tested;

```Python
def sam_AUTO(image):

  #sys.path.append("..")

  sam_checkpoint = "./sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=DEVICE)

  mask_generator = SamAutomaticMaskGenerator(model = sam,
                                              points_per_side = 32,
                                              points_per_batch = 64,
                                              pred_iou_thresh = 0.6,
                                              stability_score_thresh = 0.75,
                                              crop_n_layers=1,
                                              crop_n_points_downscale_factor=2,
                                              min_mask_region_area=100,  # Requires open-cv to run post-processing
                                              )

  img_bgr = cv2.imread(image)

  # Get the original dimensions
  height, width = img_bgr.shape[:2]
  
  new_height = int(height / 1)
  new_width = int(width / 1)
  img_bgr = cv2.resize(img_bgr, (new_width, new_height))


  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

  masks = mask_generator.generate(img_rgb)

  return masks

# Plot masks
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
        
```

The following images were captured under Creative Commons licenses, and the results are very impressive!

{{< figure class="center" src="/images/bio.png" >}}

**Conclusion**

In conclusion, the paper highlights how Segment Anything's zero-shot capabilities are a significant breakthrough in image segmentation. Although SAM can improve data labeling efficiency and accuracy, human validation remains necessary to ensure that the output aligns with the specific needs and goals of each ML project. To improve the efficiency and accuracy, it is important to combine advanced AI models and human-in-the-loop to fully unlock the potential of machine learning.

**Reference**
* https://github.com/facebookresearch/segment-anything
* https://github.com/gereleth/jupyter-bbox-widget


