+++
author = "Mohamed ABDUL GAFOOR"
date = "2022-05-05"
title = "3D UNet for Brain Tumor Segmentation"
slug = "3D UNet"
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



In this post we will discuss how to use a 3D UNet to train a deep learning model. 3D UNet is a deep learning technique used for volumetric image segmentation, which is the process of dividing a 3D image into multiple regions or segments based on their characteristics. The architecture of 3D UNet is based on the popular 2D UNet, which has been widely used in biomedical image segmentation. If you want to read the paper of 2D UNet, visit [here](https://arxiv.org/abs/1505.04597)

Similar to the 2D UNet architecture, 3D consists of an encoder and a decoder network, connected by a bottleneck layer. The encoder network down-samples the input volume by applying convolutional and max pooling layers to extract features at multiple scales, while the decoder network up-samples the feature maps to generate a segmentation map. The bottleneck layer connects the encoder and decoder networks and preserves the spatial information of the input volume.

3D UNet has been shown to achieve state-of-the-art performance in many medical image segmentation tasks, such as brain tumor segmentation, and cardiac segmentation. In this post we are going to demonstrate that 3D UNet can achieve state-of-the-art performance in various medical image segmentation tasks, including but not limited to brain tumor segmentation. The 3D UNet's capability to process 3D data makes it particularly suitable for analyzing medical images that are typically volumetric in nature.

**Brain Tumor Segmentation Challenge 2020 - BraTS2020**

BraTS2020 dataset is a collection of MRI (Magnetic Resonance Imaging) scans of the brain that have been annotated to aid in the development and evaluation of algorithms for brain tumor segmentation. The dataset was created as part of a challenge hosted by the Medical Image Computing and Computer Assisted Intervention (MICCAI) Society and consists of high-grade glioma and low-grade glioma tumor types, as well as healthy brain tissue.

The dataset includes images from multiple modalities, including T1-weighted, T1-weighted with gadolinium contrast, T2-weighted, and FLAIR (Fluid-Attenuated Inversion Recovery) sequences, as well as segmentation masks that indicate the location and type of tumors in the brain. The dataset is intended for use in the development and evaluation of deep learning algorithms for automated brain tumor segmentation, which has the potential to improve the accuracy and speed of diagnosis and treatment planning for patients with brain tumors.

Following is a visualization of a brain along 3 different projections. 

{{< figure class="center" src="/images/copy.gif" >}}

Let's take a moment to visualize the various MRI sequences, including T1, T1ce, T2,Flair and mask, that are commonly used in medical imaging to assess the brain's anatomy and pathology.
{{< figure class="center" src="/images/t1t1ce.png" >}}

The images and masks in this dataset have a shape of (155, 240, 240), where 155 is the depth and 240 represents the height and width. However, we have cropped the data to (144, 224, 224). We have opted to crop the data rather than resize it because resizing could alter the pixel values. Since we do not want any changes in the pixel values, cropping has been our preferred choice.

Following is the code to create a 3D UNet architecture;

```Python
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.models import Model
from keras.layers import Dropout
from tensorflow.keras import regularizers


def create_unet(num_layers, num_neurons, filter_size, dropout_rate, input_shape=(144, 224, 224, 4)):
    inputs = Input(shape=input_shape)

    # Encoder
    conv_layers = []
    pool_layers = []
    x = inputs
    for i in range(num_layers):
        x = Conv3D(num_neurons * 2 ** i, filter_size, activation='relu', padding='same')(x)
        x = Dropout(rate=dropout_rate)(x) # Add dropout layer
        x = Conv3D(num_neurons * 2 ** i, filter_size, activation='relu', padding='same')(x)
        x = Dropout(rate=dropout_rate)(x) # Add dropout layer
        conv_layers.append(x)
        if i < num_layers - 1:
            x = MaxPooling3D((2, 2, 2))(x)
            pool_layers.append(x)

    # Decoder
    for i in range(num_layers - 1, -1, -1):
        if i < num_layers - 1:
            x = UpSampling3D((2, 2, 2))(x)
            x = concatenate([x, conv_layers[i]], axis=-1)
        x = Conv3D(num_neurons * 2 ** i, filter_size, activation='relu', padding='same')(x)
        x = Dropout(rate=dropout_rate)(x) # Add dropout layer
        x = Conv3D(num_neurons * 2 ** i, filter_size, activation='relu', padding='same')(x)
        x = Dropout(rate=dropout_rate)(x) # Add dropout layer

    # Output
    outputs = Conv3D(4, 1, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
```

In this architecture, the input has a shape of (144, 224, 224, 4). Here, 144 represents the depth of the image, while 224 indicates its height and width as previously mentioned. Additionally, 4 represents the number of channels, with T1, T1ce, T2, and Flair serving as the four channels. Therefore, we stack all the input images to form (144, 224, 224, 4).

The model is created by first defining the input shape and then constructing the encoder and decoder layers. The encoder layers consist of multiple Conv3D layers followed by dropout layers to help prevent overfitting. The max pooling layers are also used to reduce the spatial resolution of the feature maps. The outputs of the encoder layers are saved in the conv_layers and pool_layers lists.

The decoder layers use the saved conv_layers to create an upsampling path that mirrors the encoding path. This path consists of Conv3D layers and dropout layers. The Conv3D layers have the same number of neurons as the corresponding encoding layers. The upsampling is achieved by using UpSampling3D layers to double the size of the feature maps at each step, followed by concatenation of the resulting tensor with the corresponding tensor from the encoder layer.

Finally, the output layer is a Conv3D layer with softmax activation, which generates the predicted segmentation maps. The segmentation map has pixel values of 0, 1, 2 and 4. But we have changed 4 to 3, so the pixel values were 0, 1, 2 & 3. The model is then created using the functional API of Keras, which takes the input and output layers and produces a model object that can be trained and used for making predictions.

Let us set a 4 layer network with filter size of 3 and drop out value of 0.25. See the model [here.](https://drive.google.com/file/d/17408kkQlkScTD26Z2pqe-NiwSVlX0B_2/view?usp=share_link)

```Python
model = create_unet(num_layers=4, num_neurons=16, filter_size=(3, 3, 3),dropout_rate = 0.25,input_shape=(144, 224, 224, 4))
```

Choosing a correct Metrics like **Dice coefficient** are important in deep learning to measure the performance of the model. In the context of medical image segmentation, Dice coefficient is a commonly used metric for evaluating how well the model has segmented the object of interest from the background. The Dice coefficient, also known as the F1 score, calculates the overlap between the predicted and true segmentation masks, taking into account both false positives and false negatives. It ranges from 0 to 1, where 1 represents a perfect match between the predicted and true masks. By monitoring the Dice coefficient during model training, we can see how well the model is learning to segment the object of interest. We can use this metric to compare the performance of different models, to select the best model, and to optimize the hyperparameters of the model.

In general, metrics help us to quantify the performance of the model in a meaningful way, and guide us to make improvements to the model or the data used to train the model.

```Python
def dice_coeff(y_true, y_pred, epsilon=1e-6):
    # Flatten the tensors to 2D and calculate the intersection and union
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    
    # Calculate the dice coefficient and return it
    dc = (2.0 * intersection + epsilon) / (union + epsilon)
    return dc
```

We first flatten the tensors to 2D using the tf.reshape function. Then, it calculates the intersection and union of the flattened tensors using the tf.reduce_sum function. The intersection is the number of elements that are both in the ground truth and predicted masks, while the union is the number of elements that are in either the ground truth or predicted masks. Next, the function calculates the Dice coefficient using the formula: (2 * intersection + epsilon) / (union + epsilon), where epsilon is a small number (1e-6) added to the denominator to avoid division by zero.

Similarly, we will also use **Dice Loss**, **Intersection Over Union**, **Mean Intersection Over Union**. Set the learning rate to  0.001 in Adam optimizer and loss to "categorical_crossentropy", which is used to measure the difference between the predicted probability distribution (output) and the true probability distribution (target) for a multi-class classification problem. Segmentation problems is similar to pixel-wise classification. 


**DataGenerator**

When working with large amounts of data such as 3D images and masks, it is not feasible to load all the data into memory at once, so we need a way to load the data in batches. This is where a data generator comes in. A data generator is a Python generator that yields batches of data on-the-fly during training. It loads the data from disk or other data sources, performs data augmentation and preprocessing, and then passes the data to the model for training. This allows us to efficiently work with large datasets that cannot fit into memory.

In addition to providing a way to load large datasets, data generators are also useful for data augmentation. Data augmentation is a technique where we create new data from existing data by applying transformations such as rotation, flipping, and scaling. By generating new data from the existing data, we can increase the size of our dataset and improve the generalization of the model.

Following is the data generator class in our case;

```Python
import numpy as np
import os
import nibabel as nib
from tensorflow.python.framework.ops import Tensor
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=1, dim=(144, 224, 224), n_channels=4, n_classes=4, shuffle=True, augment=True):
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.files = os.listdir(data_dir)
        self.on_epoch_end()
        self.image_datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True)
        #self.is_training = is_training   

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))
    
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        files = [self.files[i] for i in indexes]
        return self.__data_generation(files)

    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    

    def crop(self, mri_data, data_mri = True):

      # Define the cropping parameters
      start = (6, 8, 8)
      end = (-5, -8, -8)

      # Crop the MRI data using the defined parameters
      data_cropped = mri_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

      # return data_cropped
      if data_mri:
        min_ = np.min(data_cropped)
        data_cropped = (data_cropped - min_) / (np.max(data_cropped) - min_)
        data_cropped = np.round(data_cropped, 3)
        #print(data_cropped.shape)

        data_cropped = gaussian_filter(data_cropped, sigma=(1, 1, 1)) # Apply Gaussian smoothing to reduce noise
        data_cropped = data_cropped - 0.3 * laplace(data_cropped)     # Apply the Laplacian filter to sharpen the image
        
        return data_cropped

      else:
        return data_cropped

    
    def __data_generation(self, files):
       
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32) # input
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32) # output/labels


        for i, file in enumerate(files):
          
          if not file.endswith(".csv"):
            
            directory = os.path.join(self.data_dir, file)
            flair = nib.load(os.path.join(directory, file + "_flair.nii")).get_fdata()
            flair = np.moveaxis(flair, [2, 0, 1], [0, 1, 2])
            flair = self.crop(flair, data_mri = True)
                
            t1 = nib.load(os.path.join(directory, file + "_t1.nii")).get_fdata()
            t1 = np.moveaxis(t1, [2, 0, 1], [0, 1, 2])
            t1 = self.crop(t1, data_mri = True)
           
            t1ce = nib.load(os.path.join(directory, file + "_t1ce.nii")).get_fdata()
            t1ce = np.moveaxis(t1ce, [2, 0, 1], [0, 1, 2])
            t1ce = self.crop(t1ce, data_mri = True)
            
            t2 = nib.load(os.path.join(directory, file + "_t2.nii")).get_fdata()
            t2 = np.moveaxis(t2, [2, 0, 1], [0, 1, 2])
            t2 = self.crop(t2, data_mri = True)
            
            # load the mask
            mask = nib.load(os.path.join(directory, file + "_seg.nii")).get_fdata()
            mask = np.moveaxis(mask, [2, 0, 1], [0, 1, 2])
            
            mask[mask == 4] = 3 # pixel value 4 to 3
            mask = self.crop(mask, data_mri = False)

            mask = mask.astype('int')  # convert mask to integer data type
            #print("mask shape: ", mask.shape)

            num_classes = 4
            mask = np.expand_dims(mask, axis=-1)
            masks = tf.keras.utils.to_categorical(mask, num_classes)

            # create stack for images and one-hot for masks..
            images = np.stack([flair, t1, t1ce, t2], axis=-1)
      
            if self.augment:
              # apply random rotation and flip
              seed = np.random.randint(1, 100)
              datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True)

              for k in range(len(images)):
                images[k] = datagen.random_transform(images[k], seed=seed)
                masks[k] = datagen.random_transform(masks[k], seed=seed)
                masks = masks.astype(int) # convert to int type so we will have 0 and 1

              X[i,] = images
              y[i,] = masks

            else:
              X[i,] = images
              y[i,] = masks

        return X, y
```

* The __len__ method: return the length of the object, which is typically the number of batches in the dataset.
* The __getitem__ method: takes an index as input and returns the data associated with that index. The index corresponds to a specific batch of data in the dataset. 
* **on_epoch_end** method: the indices of all the files in the dataset are created using np.arange(len(self.files)). If shuffle=True, the indices are then shuffled using np.random.shuffle(self.indexes). This ensures that during each epoch, the data is presented in a different order, which can help prevent the model from overfitting to any specific patterns in the data.

* crop method: crop the data and as well as apply Gaussian smoothing to reduce noise and Laplacian filter to sharpen the image.

Moreover, we have applied data augmentation technique and made sure masks have **int type** so we will have 0 and 1 in the mask after one-hot encoding. Additionally, the ultimate dimensions of both the image and mask are (1, 144, 224, 224, 4), with 1 representing the batch size, 144 signifying the depth, and 224 denoting the height and width respectively. The value 4 denotes the number of channels or classes, depending on whether the data pertains to the image or the mask.

To simplify the process, we have separated the training and validation data into distinct folders.

```Python
train_dir = "/path_to_train_data/train"
val_dir = "/path_to_val_data/validation"

batch_size = 1
shuffle = True
```

The next step involves creating two instances of a custom DataGenerator class, one for training data and another for validation data. The DataGenerator class generates batches of data for training or validation purposes, with optional data augmentation and shuffling.

```Python
training_generator = DataGenerator(data_dir= train_dir, batch_size = batch_size, shuffle = True, augment = True)
validation_generator = DataGenerator(data_dir= val_dir, batch_size = batch_size, shuffle = True, augment = True)
```
Now we can start the training;

```Python
checkpoint_filepath = 'path_to_the_checkpoint/where_you_have_to_store/'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    save_best_only = True)
    
history = model.fit(training_generator,epochs=4,
                    validation_data=validation_generator,
                    use_multiprocessing=True,callbacks=[checkpoint],
                    workers=6, verbose=1) 
```
After 4 epoch, the model starts to converge and following is the result;

```Python
loss: 0.0469 - accuracy: 0.9876 - dice_coeff: 0.9788 - iou: 0.9588 - mean_iou: 0.9023 - val_loss: 0.0417 - val_accuracy: 0.9879 - val_dice_coeff: 0.9810 - val_iou: 0.9629 - val_mean_iou: 0.9194
```

Save the model;
```Python
model.save("/path_to_save/model2.h5")
```

Let us see how our model perform on the validation data;
```Python
nextbatch = next(iter(validation_generator))
X_, y_ = nextbatch
```

Load the saved model
```Python
model_path = "path_to_the_model/model2.h5"
```

Define the custom objects; 
```Python
custom_objects = {'loss': "categorical_crossentropy", 'dice_coeff':dice_coeff, 'iou': iou, 'mean_iou': mean_iou}
```

If you train with the custom functions, you have load with the custom function. 

```Python
modelbrats = load_model(model_path, custom_objects=custom_objects)
```
Load the saved weights

```Python
modelbrats.load_weights(model_path)
```

Finally, use the loaded weights to make predictions;

```Python
y_pred = modelbrats.predict(X_)
```

Let us visualize for an arbitary slice;
```Python
slice = 75
y_pred_1 = y_pred[0, :, :, :, 0][slice]
y_1 = y_[0, :, :, :, 0][slice]
X_1 = X_[0, :, :, :, 0][slice]
```

If we check the **np.unique(y_pred_1)**, we get the following result.

```Python
array([0.01908756, 0.01910349, 0.01917317, ..., 0.9999997 , 0.9999998 ,
       0.99999994], dtype=float32)
```
We can notice the predicted pixel values are not 0 or 1. So let us convert them to intergers using **np.around(y_pred_1)**

```Python
y_pred_1 = np.around(y_pred_1)
```
Let us visualize ground truth, prediction and the brain for few slices.. Following is the code;

```Python
import matplotlib.pyplot as plt

start_index = 70
end_index = 74

fig, axs = plt.subplots((end_index - start_index + 1), 3, figsize=(15, 30))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
fig.patch.set_facecolor('white')  # set the color of the coordinate axis to white

for i in range(start_index, end_index+1):
    axs[i-start_index, 0].imshow(np.round(y_pred[0,i, :, :, 0]), cmap='bone')
    axs[i-start_index, 0].set_title('y_pred {}_Predicted'.format(i), color='blue')
    axs[i-start_index, 1].imshow(y_[0,i, :, :, 0], cmap='bone')
    axs[i-start_index, 1].set_title('y_ {}_Ground Truth'.format(i), color='blue')
    axs[i-start_index, 2].imshow(X_[0,i, :, :, 0], cmap='bone')
    axs[i-start_index, 2].set_title('X_ {}_Brain'.format(i), color='blue')

plt.show()
```
The figure below shows the predicted mask along with the ground truth. 

{{< figure class="center" src="/images/tumorspred.png" >}}

**Conclusion**

3D UNet is better than 2D UNet in certain scenarios because it can capture more spatial information and provide better segmentation accuracy. 2D UNet works well for segmentation tasks on 2D images. However, when it comes to volumetric data such as medical images of organs, tissues, and lesions, the 2D U-Net may not perform as well because it cannot capture the spatial context of the data in the z-axis (depth). The 3D UNet is designed to handle volumetric data and captures spatial information in all three dimensions (x, y, and z). It can capture the context of the 3D image by taking into account the neighboring slices in the z-axis. As a result, it can produce more accurate segmentations than the 2D U-Net in volumetric data.




