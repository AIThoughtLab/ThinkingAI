+++
author = "Mohamed ABDUL GAFOOR"
date = "2022-05-18"
title = "Marching Cube Algorithm"
slug = "mc"
tags = [
    "Marching Cubes",
    "3D Reconstruction"
]
categories = [
    "Computer Graphics",
    "Computational Geometry"
]

+++

**Marching Cube Algorithm - an overview:**

The Marching Cube Algorithm is a well known algorithm in computer graphics and scientific visualization for creating a 3D surface mesh from a 3D scalar field. 
The input of marching cube algorithm is a 3D voxels or 3D coordinate points and the output is a triangular mesh that represents the isosurface of the scalar field.

The basic idea of the algorithm is to divide the input volume into a regular grid of cubes, with each cube having 8 corners. If all 8 corners of a cube have positive numbers or all 8 corners have negative numbers, then the whole cube will either be above or below a certain surface. When this happens, **no triangles will be created**. Technically 256 possible configurations (2 power 8). However, several of these configurations are equivalent to one another. Therefore, only 15 unique cases exist, which are displayed here:

{{< figure class="center" src="/images/mc.png" >}}


Then, for each cube, the algorithm determines the subset of the cube's faces that are intersected by the isosurface of the scalar field. This information is used to generate a set of triangular facets that approximate the surface of the isosurface. The output of the algorithm is a triangular mesh that can be rendered using standard graphics techniques. The mesh is guaranteed to be topologically correct, meaning that it does not contain any holes or self-intersections, and it approximates the isosurface of the scalar field to within a specified tolerance.

In this blog we will use a Python library called **mcubes**, which provides tools for generating isosurfaces from volumetric data using the Marching Cubes algorithm. This library is useful for extracting surfaces from 3D medical imaging data, in our case 3D MRI (brats2020)

To utilize the mcubes library in Python, it can be installed by executing the command **pip install --upgrade PyMCubes**. This will download and install the latest version of the library, which can then be imported and utilized in Python scripts or notebooks for generating isosurfaces from volumetric data using the Marching Cubes algorithm. Github page of PyMCubes can be found [here.](https://github.com/pmneila/PyMCubes)


When dealing with the brats2020 dataset, the mcubes library can be utilized for reconstructing the 3D surface of brain tumors. This is done by processing the **seg.nii** data from training/validation sample or our own prediction. By utilizing the mcubes library, it is possible to generate accurate and visually appealing representations of the brain tumor surfaces, which can be analyzed and used in a variety of research and clinical applications. In brats2020 dataset, the segmentation masks have pixel values of 0, 1, 2 or 4, and we can identify 3 main type of tumors, namely, Whole tumor (WT), Tumor Core (TC) or Enhancing Tumor (ET). 

Following is the code to reconstruct the 3D surfaces;

```Python
import mcubes
import nibabel as nib
import numpy as np

mask = nib.load('seg.nii')

mask = np.array(mask.dataobj)
print(np.unique(mask)) # Make sure the unique values

mask_WT = mask.copy()   # WT = Whole tumor 
mask_WT[mask_WT == 1] = 1
mask_WT[mask_WT == 2] = 1
mask_WT[mask_WT == 4] = 1

mask_TC = mask.copy()   # Tumor Core 
mask_TC[mask_TC == 1] = 1
mask_TC[mask_TC == 2] = 0
mask_TC[mask_TC == 4] = 1

mask_ET = mask.copy()   # Enhancing Tumor
mask_ET[mask_ET == 1] = 0
mask_ET[mask_ET == 2] = 0
mask_ET[mask_ET == 4] = 1

# possible smooth functions;
# mcubes.smooth_constrained
# mcubes.smooth
# mcubes.smooth_gaussian

mask_list = [mask_WT, mask_TC, mask_ET]

for i,mask in enumerate(mask_list):
  smoothed_nii = mcubes.smooth_gaussian(mask)
  vertices, triangles = mcubes.marching_cubes(smoothed_nii, 0)

  mcubes.export_obj(vertices, triangles, f'smooth_{i}.obj')
```

Following is the 3D reconstructed model using mcubes library (Vertices: 24,924 and Triangles: 49,844), you can download the **.obj** files [here.](https://drive.google.com/drive/folders/1722pJRz2rbrzA9444djyHERRwrTH0xO0?usp=share_link)

{{< figure class="center" src="/images/model.png" >}}

There are many benefits of using 3D reconstruction in scientific and engineerinh domain because 3D reconstruction allows us to visualize objects and environments in 3D space. This can help us better understand complex structures and systems. Similarly, by creating accurate 3D models of objects and environments, we can make precise measurements and perform detailed analysis. This can be particularly useful in scientific research, engineering, and medical fields.

