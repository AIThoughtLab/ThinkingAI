+++
author = "Mohamed ABDUL GAFOOR"
date = "2020-12-26"
title = "Human 3D Pose Reconstruction"
slug = "pose"
tags = [
    "Computer Vision",
    "3D Reconstruction",
    "Pose Estimation"

]
categories = [
    "Artificial Intelligence"
]

+++

**Human 3D Pose Reconstruction**

This blog focuses on performing 3D reconstruction of human poses using multiple views captured by a calibrated system. To achieve this, a deep neural network called OpenPose is used to estimate the location of the joints in the human body. The OpenPose model has been trained to recognize 25 different joints in the human body, such as the head, shoulders, elbows, hips, knees, and ankles. These joints are critical for determining the posture and movement of a person.

In 3D reconstruction, a calibrated system means that the parameters such as focal length, sensor information, and lens distortion are known in advance. This enables us to accurately measure the position and orientation of each camera, allowing for precise multi-view reconstruction of the human pose. Accurately estimating the joint locations of a person in 3D space has numerous applications in fields such as motion capture, virtual reality, and human-computer interaction.

{{< figure class="center" src="/images/joins.png" >}}

In this blog, Structure from Motion (SfM) will be used to complete the task of 3D reconstruction. SfM is a technique used in computer vision to estimate the 3D structure of a scene from a sequence of images captured by a moving camera. In SfM, the camera poses are estimated first, followed by the reconstruction of the 3D scene. This is done by analyzing the displacement vector, or optical flow field, between the images captured from different viewpoints. The goal of SfM is to estimate the 3D structure of a scene by analyzing the changes in camera poses and corresponding image features. For example, like in the figure below, there are several images that are taken from different perspective and reconstructed to create a 3D object. 

{{< figure class="center" src="/images/sfm.png" >}}

The central idea of this approach is to identify the critical points within a sequence of image data or a movie. By tracing these key points, we can then determine both the location of the cameras and the 3D coordinates of the identified points. With this information in hand, we are able to reconstruct the 3D geometry of the scene. Essentially, this technique enables us to extract valuable data from 2D images and convert it into a 3D representation of the scene.

**2D key-points matching**
Here the objective is to read the 2D joints detected by openPose and visualize it on the corresponding video to ensure the 2D data is correct. As we said there are 25 joints and the 2D coordinates of the output is in the form of; xJ1 yJ1 rJ1 xJ2 yJ2 rJ2 xJ3 yJ3 rJ3....... xJ25 yJ25 rJ25.
Here xJ1 yJ1 are the x & y coordinates of the joints and the r is the reliability score. For example, following is the dataset of **squat_1_0.0.txt** [(link)](https://drive.google.com/drive/folders/1ig_AGpMrr6RsRXpzJ9etxeNkGbWNJbnHcI?usp=share_link). Here 333.318, 98.4255 are the x & y coordinate, and the 0.906826 is the reliability score.

{{< figure class="center" src="/images/short.png" >}}

The following function was used to extract the 2D coordinate positions. It takes three argument; the location of the text file where the coordinate information are saved, video path and the name of the output file to be generated. 

```Python
def keyPoints2D(path, videoPath, output):
  with open(path) as f:
    lines = []
    for line in f:
      lines.append(line)

    cap = cv2.VideoCapture(videoPath)
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video = cv2.VideoWriter(f'{output}.avi',fourcc, int(cap.get(5)), (int(cap.get(3)),int(cap.get(4))))

    count = 0
    counter = 0
    bad_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
          coordinates = []
          
          for i in lines[counter].split():
            if i <= str(1.0):
              pass
            else:
              coordinates.append(int(float(i)))

          pairs = list(zip(coordinates[::2], coordinates[1::2]))
          pts = np.array(pairs)

          for p in pts:
            frame_ = cv2.circle(frame,tuple(p),2,(0,255,255), thickness=-1)

            if set(p) == set(list(pts[-1])):
              video.write(frame_)
              #cv2_imshow(frame_)

          counter +=1

        else:
            cap.release()
            bad_frame +=1
            break
        #print("Number of bad frames: ", bad_frame)

      #cv2_imshow(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()
```

We also define another function to get the data. In our case this function will choose only the **"squat"** text data or related video files and finally, we will execute it as follow;

```Python
def getData(path):
  data_list = []
  for data in sorted(os.listdir(path)):
    if data.startswith('squat'):
      data_list.append(data)
  return data_list
  
vidList = getData(videoPath)
textList = getData(textPath)

for i in range(len(vidList)):
  fileName = ''.join(vidList[i].split(".avi"))

  textPath_ = textPath+textList[i]
  videoPath_ = videoPath+vidList[i]
  keyPoints2D(textPath_, videoPath_, fileName)
```

In the following video, we can see that the 2D points are embedded.

{{< video src="/images/video.mp4" type="video/mp4" preload="auto" >}}


Next we use the configuration file of cameras to shift the origin of 3D joints (pre-estimated) having reference on camera 0 to reference of other cameras (1 2 3) and project the 3D joints back to image. The purpose is to make sure we understand the translation, rotation between camera and the projection matrix. So let us use the configuration file of cameras to shift the origin of 3D joints from the reference zero camera to the another reference cameras, may be 1 or 2 and project the 3D joints back into the image. The below is the output of the origin shift matrix from 0 to 1. You can see this is a 3x4 matrix. 

{{< figure class="center" src="/images/metrix.png" >}}

We can use this matrix along with the intrinsic parameter of the cameras, the 2D pose on view 2 for example, 3D pose after triangulation (reference at view 0), and retrieve the 3D points at view 0. The we can recall the origin shifting method which was define like below to shift the origin from view 0 to view 1or 1 to 2 etc.

```Python
def originShift(point,P):
    point = np.append(point, np.array([1]))
    point = np.dot(P,point)
    return point
```

Finally after the normalization, it can be projected into 2D point on view2 (or any other), using the intrinsic of camera 2 (or any other). The below is a generated video by the script. The video shows the OpenPose (by big color circle) & the projected point after origin shifting with smaller red circle. 

{{< video src="/images/video1.mp4" type="video/mp4" preload="auto" >}}

**3D pose reconstruction**

With the understand of camera projection matrix, let us estimate the 3D joints from the corresponding 2D joints between views and camera projection matrix;
The following steps must be taken in order to reconstruct the 3D coordinates. 

1. We must estimate the fundamental matrix from key points in two images
2. If we have the fundamental matrix and the camera intrinsic matrix, we can calculate the essential matrix. 
3. If we have the essential matrix, we can calculate the rotation and the translation using single value decomposition (SVD) .
4. We can compute the projection matrices using the rotation matrices and the camera calibration matrices. 
5. With all these information, we can **cv2.triangulatePoints** to get the 3D coordinates (world coordinates). 

The following code snippet helps us to reconstruct the 3D world coordinates.

{{< figure class="center" src="/images/matrix.png" >}}

The following video is a reconstructed one between the view 0 & view 1. In this reconstruction you can notice that we have used P01, when we reconstruct the P2 projection matrix. 

{{< video src="/images/video2.mp4" type="video/mp4" preload="auto" >}}

References:

1. http://www.cse.psu.edu/~rtc12/CSE486/lecture25.pdf 

2.  https://stackoverflow.com/questions/18018924/projection-matrix-from-fundamental-matrix 

3.  https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#triangulatepoints 

4.  https://www.youtube.com/watch?v=kq3c6QpcAGc&t=543s 

