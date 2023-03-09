+++
author = "Mohamed ABDUL GAFOOR"
date = "2023-01-02"
title = "Biological Image Analysis using ImageJ"
slug = "IJ"
tags = [
    "ImageJ",
    "Data Analysis"
]
categories = [
    "Computer Vision",
    "Image Analysis"
]

+++

**How to use ImageJ for biological image analysis**

Image analysis is a critical step in many research fields, including biology, medicine, and materials science. One widely-used software for image analysis is ImageJ, which is a powerful, user-friendly program developed by the National Institutes of Health (NIH).

ImageJ allows users to analyze digital images by performing a range of quantitative measurements, such as area, intensity, and particle size. It also has a wide variety of image processing and manipulation tools, including filters, segmentation, and thresholding. These features make it a versatile tool for many different applications, from analyzing protein expression in cells to measuring the size and shape of nanoparticles.

This post will demonstrate the use of ImageJ for basic image processing. The necessary data can be downloaded from the following link [here.](https://drive.google.com/drive/folders/1NwXAxNLAk5jxwxqhhx_GrWtHE4DuaKPE?usp=share_link)

**Salt and pepper noise**

Salt and pepper noise is a type of image distortion that occurs due to errors in image acquisition or transmission. It is characterized by the appearance of white and black dots, which can significantly reduce the clarity and quality of an image.

Salt and pepper noise is caused by the presence of defective pixels in the image sensor or errors in the transmission of image data. It is named after the white and black speckles that resemble grains of salt and pepper. How do we remove such noises?

The image below displays scattered white and black dots, with white dots appearing in black regions (salt) and black dots in white regions (pepper). The removal of this noise is essential in image processing tasks. In ImageJ, to eliminate salt and pepper noise, the image must first be converted into an 8-bit grayscale image, followed by the application of a median filter with a suitable radius. In our case a radius of 4 pixels.

{{< figure class="center" src="/images/sp.png" >}}

**Uneven illumination**

Cropping is a viable option for dealing with uneven illumination. Alternatively, in our case, we can apply Gaussian blur to the image in ImageJ after converting it to an 8-bit image. By adjusting the sigma radius parameter through trial and error, we can obtain the image's background. Once we have obtained the background, we can subtract it from the original image to eliminate the uneven illumination. Another approach is to calculate the mean intensity of the Gaussian-blurred image and use the Calculator Plus functionality in ImageJ to divide the image from the background by setting mean intensity. The resulting image is shown below.

{{< figure class="center" src="/images/gauss.png" >}}

**Approach to denoise**

ImageJ can be used to reduce noise in images by utilizing binary masks. Firstly, we must load the TIF file of the image into ImageJ and create a duplicate copy to avoid making any changes to the original file. If needed, we can utilize Enhance Contrast to normalize all the images in the stack with a different percentage of saturated pixels. After that, we should apply a smoothing function (such as Gaussian filtering) to all the images. This method removes noise while also slightly blurring the image. Both of these functions can be found under the "Process" tab in ImageJ.

Next, we can use the Thresholding technique to further process the image. In ImageJ, there are various Thresholding techniques available such as IsoData, Otsu, Yen, etc. It is recommended to try different Thresholding methods to determine which one works best with our specific biological data. Once a suitable Thresholding method is selected, we can create a binary mask. Based on the output mask, we can apply erosion and/or dilation to remove or add pixels to the object boundary. Finally, we can utilize the Image calculator option in ImageJ to combine the mask and the original image, which will result in a reduced noise image. In addition to this, we can also test the built-in Despeckle denoising functionality in ImageJ. This functionality uses a median filter to replace each pixel with the median value in its 3 x 3 neighborhood, which provides better noise reduction.

In addition to the aforementioned techniques, we can also explore the FFT functionality in ImageJ for noise reduction. The FFT bandpass filter is a useful tool for reducing noise in images by removing high spatial frequencies and low spatial frequencies. Since the samples in our case are fluorescently labeled and the images are taken over a period of time, there may be intensity attenuation over time. Therefore, photobleaching correction should be performed to restore the correct intensity. ImageJ offers a plugin for bleach correction with different correction methods, such as simple ratio, exponential fitting, and histogram matching methods. The following image demonstrates the effect of bleach correction on the stack (14/26), with and without correction.

{{< figure class="center" src="/images/bleach.png" >}}

We can also explore the possibility of improving the image through deconvolution techniques, which require a Point Spread Function (PSF) generator plugin in ImageJ. If the necessary parameters are available, this can be another effective approach for denoising.
Recently, advanced denoising techniques have been developed, such as Noise2Void - a deep learning-based method that learns the characteristics of noise images and denoises them with minimal human intervention. A plugin for using Noise2Void is available on the deepimagej website for ImageJ. We can select either the Noise2Void denoising model or CARE deconvolution. The screenshot below shows the DeepImageJ interface.

{{< figure class="center" src="/images/deepj.png" >}}

**How would you quantitatively measure the quality of a denoising result?**

Quantitatively measuring the quality of denoising is important to ensure that the denoising process has been effective and has not introduced any artifacts or distortions in the image. It provides a standard and objective way to evaluate the performance of different denoising methods and to compare their effectiveness. ImageJ provides a plugin that enables quantitative measurement of noise quality. Metrics such as Signal-to-Noise Ratio (SNR), Peak Signal-to-Noise Ratio (PSNR), Root-Mean-Square Error (RMSE), or Mean Squared Error (MSE) can be used. To compare the noise in the original image (2_Noisy.tif) with the restored image (3_Restored.tif), we can use the SNR plugin. A good-quality confocal fluorescence image usually has a SNR/PSNR ranging from 20 to 40dB. Below is a table displaying the results.

{{< figure class="center" src="/images/snr.png" >}}

**Conclusion**

In my opinion, ImageJ is a powerful and versatile image processing software that can be used for a wide range of tasks, including noise reduction, image segmentation, and quantitative analysis. Its user-friendly interface and extensive collection of plugins make it accessible to both novice and experienced users. I am impressed with ImageJ's wide range of image processing capabilities, including a variety of noise reduction techniques, thresholding, and deconvolution. It also allows for quantitative measurements of image quality, which is essential for ensuring that the output of image processing tasks is accurate and reliable. In summary, I believe that ImageJ is a valuable tool for anyone working with digital images, particularly in the fields of biology and medicine.





