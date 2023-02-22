+++
author = "Mohamed ABDUL GAFOOR"
date = "2021-09-05"
title = "3D Reconstruction Based on Medical Imaging For Personalized Surgical Procedures In Cardiac Rhythmology"
slug = "Cardiac Rhythmology"
tags = [
    "Deep Learning",
    "Machine Learning",
    "Virtual Reality",
    "3D reconstruction"
]
categories = [
    "Artificial Intelligence"
]

+++

_This is a short summary of my master thesis on VR at the Institut Polytechnique de Paris (IP Paris)_

**Abstract**: 
Virtual Reality (VR) technology plays a vital role in Science and Engineering for immersive visualization and is a new interactive paradigm. The recent advancement
in the processing power of computers have enabled tremendous opportunities for the increased use of VR technology in medicine and surgery. Preoperative planning
of Catheter Ablation (CA) Therapy in the VR environment is an important immersive application not only to minimize the risk of intricacy during surgery but also
to reduce the operating time, a method of training for novice surgeons and for patient education. In this study, we demonstrate a 3D reconstruction of Left Atrium
(LA) from Computerized Tomography (CT) images and a complete immersive visualization and navigation of catheter in order to facilitate the surgical procedures.



**Background**: Cardiovascular Diseases (CVDs), also known as heart diseases that affect the heart and the blood vessels of millions of people each year. According to the World Health
Organization, approximately 17.9 million people died from CVDs in 2019, which represents 32 % of all the global deaths1 . Findings from Global Burden of Disease
Study 2017 shows that even though there are cost-effective interventions available to prevent deaths from CVDs, for example, lower blood pressure or cholesterol, the
mortality rate has significantly increased since 2007 worldwide2 . In contrast, CVDs in high-income countries have declined substantially compared to the low-income
countries over the past half-century. Nevertheless, in high-income countries this long term trend has stagnated in recent years or even increased in younger populations.


**Project Scope and Motivation**:
The main scope and the motivation behind this project is to facilitate the surgical procedure of the CA therapy for left atrial fibrillation. More specifically, the work is focused on patient-specific surgical planning where the patient’s 3D reconstructed heart model is used for the rehearsal purpose using the VR technology before the real surgery. It is therefore intended to design, develop and evaluate a new preoperative planning tool for the CA therapy for AF. The proposed hypothesis is that the developed VR application would allow the surgeons to take the right decision before the surgery and helps to increase the safety and the efficiency in the operating room. Moreover, this research can also be extended to the educational training program for young surgeons and as well as for the patient education. Here we will use three state of the art technologies; namely deep learning, computer vision and the virtual reality. At the end of the project, a working VR application will be developed to facilitate the surgical procedure that mainly involves immersive visualization and the navigation of catheter.

**Research Objectives**:

The objectives are as follows:
* OBJECTIVE : 1 To apply U-Net architecture and/or a threshold techniques to segment CT images of left atrium.
* OBJECTIVE : 2 To apply marching cubes algorithm for the 3D reconstruction of the segmented images.
* OBJECTIVE : 3 To use Unity game engine for the simulation of immersive visualization and the navigation of catheter.



**Image Segmentation Using U-Net**

U-Net is build upon a fully convolutional network, with certain modifications to its base architecture, which are more suitable for biomedical imaging applications.
It receives an image as an input and produce an output image with the same size as the original image.

{{< figure class="center" src="/images/unet.png" >}}


The figure above shows that the lowest resolution is 32x32 pixels. Blue boxes represent multi-channel feature map. In each boxes the number on top of it denotes
the channel number. For example, the input image is a 1-channel. The white boxes are copied feature map to the decoding layers. The outupt segmentation map contains two classes (foreground and background). The blue arrows represent the convolution operations and followed by a non-linear activation function. The Encoder (which is in the left side) part consists of repeated application of two
3x3 convolutions passing through ReLU activation functions, then 2x2 MaxPooling and the Dropout layer. Number of filters will increase in each stage and dimensionality of the features will be reduced using the pooling layer. In the Decoder (which is in the right side) each step consists of upsampling of the feature map, then deconvoluted by a 2x2 transpose which helps to halves number of feature channels. Also the network consists of concatenation with the corresponding feature map from the encoder path (contraction) to the decoder path (expension). In the final stage 1x1 convolution is used to get the desired output classes. Following figure shows the ground truth and predicted mask.

{{< figure class="center" src="/images/gt.png" >}}


**Image Segmentation Using Thresholding and follow up 3D reconstruction**

Thresholding is a basic segmentation technique used in many Computer Vision (CV)
tasks. In this technique, an image pixels are grouped depending on their intensity
values, which makes the images easier to analyze. A gray scale image converted into
binary using some conditions. For example, if the pixel value greater than a certain
value, we set to 1, otherwise we set to zero. i.e,

{{< figure class="center" src="/images/eq.png" >}}

Where T is here the appropriate thresholding value. The below figure is an example, in which a thresholding technique was used to segment the left atrium. The
image was converted into a gray scale 8-bit image and a thresholding technique was applied. The histogram of the image in Figure 2.9 show that the image is segmented
using thresholding value of above 185. Note that the image pixel values are set to 0 to 255. This means all the values below 185 removed and only the values above will remain. This helps to remove the unnecessary regions around the image and to focus on the region of interest. This technique helps to distinct the chambers of heart.

{{< figure class="center" src="/images/imageMask.png" >}}

Following histogram shows the distribution of the pixel values for a given image.
{{< figure class="center" src="/images/hist.png" >}}


we segment the image using the information about the contrast material injection before the CT scan. However, we will find many other vessels and bones (ribs, vertebral) appear in the segmented image.
Hence the image must go through further refining process. Removing the bones and vessels can be easily achieved by using region labeling (region extraction) algorithms. Moreover, the catheter wires can be removed manually if necessary or else this can be removed when we do the cleansing process of the calibers. During the thresholding because of the noise sometimes the holes might appear
in the middle of the structures. These holes must be removed before we apply the cleaning process. Otherwise, it may bias the caliber computation. The figure below shows before and after removing the bones from the initial segmented image.

{{< figure class="center" src="/images/bone.png" >}}


Formation of tinny holes after thresholding, due to noise can be found here.

{{< figure class="center" src="/images/noise.png" >}}


The developed method for caliber computation consists of two main steps namely; candidates selection and distal reconstruction. We implement the same technique for the left atrium. The figure below shows the segmented slices (uncleaned) and the corresponding cleaned segmented region of interests. 

{{< figure class="center" src="/images/cal.png" >}}


Similarly, the figure below shows the caliber computed image. After this step, again thresholding should be applied to the image to remove the vessels completely.

{{< figure class="center" src="/images/cal1.png" >}}

The following figure shows the full 3D rendered heart model in color heatmap with calibers and on the right side of the figure the corresponding heart model where are vessels are removed.

{{< figure class="center" src="/images/rendered.png" >}}


**Marching Cube algorithm**

In Marching Cube algorithm, the input data is a voxel information, ie, a scalar field //(V = f(x, y, z)//). It could be binary or not. If it is not a binary, the Marching Cube
algorithms needs an additional parameter in order to distinct the sample points inside or outside the surface. For the binary model, the interior or the exterior points
are separated and would obtain a surface like. If the model is not binary, we create a isosurface joining all the points from given isovalue. The figure below shows the points joining by the isovalues to form level surfaces (in this case level lines) that maps from $$R^{3} \Rightarrow R$$

{{< figure class="center" src="/images/march.png" >}} 


Hence the basic idea is that, in the Marching Cube algorithm the cube of size 2x2x2 traverses along the sample and in each cube the algorithm creates set of triangles. The output surface has to satisfy the conditions; such as it should be able to separate interior points with the exterior points and should be closed and orientable (surface normal vector). The figure below shows the closed and the orientable surface created with the Marching Cube algorithm.

{{< figure class="center" src="/images/cube.png" >}} 

Finally the figures below shows the left atrium;

{{< figure class="center" src="/images/left.png" >}} 

and a complete 3D heart model (ready to be deployed in the Unity). 

{{< figure class="center" src="/images/3dheart3.png" >}}


**Catheter simulation**

A simulation of a catheter motion in a VR environment is a highly complex task. However, this is an important step in realizing the complexity when navigating
through the femoral vein to the left atrium. In this project we tested many ways to reach this and finally one method worked out well. In this method we create
many spheres as body parts of the catheter, and also we set a minimum distance between the body parts, i.e spheres, and moreover speed and the rotational speed are
set as convenient. Moreover we create a body prefabs in order to instantiate every prefabs, this step is good if we decided to include many body parts as possible. We
also create a camera game object so that the first sphere carries the camera game object to record the scene in front of it. Because of the homogeneous and the isotropic
nature of the sphere game object, rotation of the sphere (which carries the camera) is not going to be visible in the scene. To overcome, we create another game object to as a pointer that is attached to the first sphere. 

Now once completed catheter successfull, we are ready to deploy it in the Unity evvironment along with the reconstructed heart. 

{{< figure class="center" src="/images/load.png" >}}

To see the navigation, check out the video [here](https://drive.google.com/file/d/1j9OknpDit5P67tmhjr3F2mkblOGJnWji/view?usp=share_link) or to see the turn/snap turn check out [here.](https://drive.google.com/file/d/14KreEnKciH000x1wcRRG2NuOSKBail1Y/view?usp=share_link)




