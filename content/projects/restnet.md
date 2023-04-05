+++
author = "Mohamed ABDUL GAFOOR"
date = "2023-04-05"
title = "Building a Segmentation Model: Creating a UNet architecture from ResNet50 as encoder"
slug = "restnet"
tags = [
    "UNET",
    "Machine Learning",
    "Deep Learning",
    "Mixed Reality"
]
categories = [
    "Artificial Intelligence", "Mixed Reality"
]

+++


In this post, I will be discussing a process of creating a UNet architecture where the encoder is sourced from the **Resnet50**. This technique is highly valuable, particularly when working with limited data, as it enables us to leverage the benefits of a pre-trained Resnet50 network. The UNet is an incredibly popular deep learning architecture for segmentation tasks and has become a go-to solution for many practitioners in this field. If you are interested in learning more about the original 2D UNet paper, I highly recommend referring to the source [material.](https://arxiv.org/abs/1505.04597)

In the 2D UNet architecture, there are two main components: an encoder and a decoder network, which are linked by a bottleneck layer. The encoder network extracts features at multiple scales by using convolutional and max pooling layers to down-sample the input volume. On the other hand, the decoder network up-samples the feature maps to create a segmentation map. The bottleneck layer serves as the connection between the encoder and decoder networks, and it retains the spatial information of the input volume. By using this architecture, the 2D UNet can effectively segment images while preserving the spatial relationship between different regions of the image.

To accomplish this task, we will be utilizing a database specifically designed for hand gesture recognition. The dataset, which can be downloaded from the following link [here](https://sun.aei.polsl.pl//~mkawulok/gestures/), pertains to the HGR1 type of data. This dataset includes both original images and skin masks, and consists of 899 images from 12 individuals performing 25 unique gestures. The image dimensions in the dataset vary, ranging from 174x131 up to 640x480 pixels. The backgrounds in the images are uncontrolled, as are the lighting conditions. The masks have 4 channels, the last one is **alpha** channel. We can ignore the alpha channel safely. This is because the alpha channel encodes the transparency of the image and is not relevant for image segmentation.
Masks have 0 or 255 values and all three channels are identical, meaning, they have been duplicated across all three channels. In this case, we can treat the problem with only one channel and convert the masks to grayscale.


As a first step, import the necessary libraries;
```Python
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from google.colab.patches import cv2_imshow
from keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from keras.utils import to_categorical
from tensorflow.keras import layers
```

Set the path to the image and the mask folder.

```Python
image_folder = "./original_images/"
mask_folder = "./skin_masks/"
```

- We will create list of filenames for images and masks.
- Split the filenames into training and testing sets.
- Create the full paths to the training and testing images and masks

```Python
image_filenames = sorted(os.listdir(image_folder))
mask_filenames = sorted(os.listdir(mask_folder))

#  Split the filenames
train_image_filenames, test_image_filenames, train_mask_filenames, test_mask_filenames = train_test_split(image_filenames, mask_filenames, test_size=0.2)

#Create the full paths
train_image_paths = [os.path.join(image_folder, filename) for filename in train_image_filenames]
test_image_paths = [os.path.join(image_folder, filename) for filename in test_image_filenames]
train_mask_paths = [os.path.join(mask_folder, filename) for filename in train_mask_filenames]
test_mask_paths = [os.path.join(mask_folder, filename) for filename in test_mask_filenames]
```
We will display few images and the corresponding masks;
```Python
def plot_images_and_masks(image_paths, mask_paths):
    fig, ax = plt.subplots(len(image_paths), 2, figsize=(5, 5))

    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load image and mask
        img = plt.imread(image_path)
        img = normalize_array(img)

        mask = plt.imread(mask_path)
        mask = normalize_array(mask)

        # Plot image and mask side by side
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(mask, cmap='gray')

        # Set title
        ax[i, 0].set_title(f"Image {i+1}")
        ax[i, 1].set_title(f"Mask {i+1}")

        # Remove axis ticks
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

plot_images_and_masks(train_image_paths[0:5], train_mask_paths[0:5])
```
Following is the visualization of 5 images and masks. 

{{< figure class="center" src="/images/handsmasks.png" >}}

**Data Generator**

When working with large amounts of data, it is not feasible to load all the data into memory at once, so we need a way to load the data in batches. This is where a data generator comes in. A data generator is a Python generator that yields batches of data on-the-fly during training. It loads the data from disk or other data sources, performs data augmentation and preprocessing, and then passes the data to the model for training. This allows us to efficiently work with large datasets that cannot fit into memory.

In addition to providing a way to load large datasets, data generators are also useful for data augmentation. Data augmentation is a technique where we create new data from existing data by applying transformations such as rotation, flipping, and scaling. By generating new data from the existing data, we can increase the size of our dataset and improve the generalization of the model.

Following is the data generator class in our case;
```Python
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, image_size=(224,224), batch_size=16, img_channels=3, mask_channels = 1, shuffle=True, augmentation=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()
        self.img_channels = img_channels
        self.image_size = image_size
        self.mask_channels = mask_channels
        self.augmentation = augmentation

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths = [self.image_paths[k] for k in indexes]
        mask_paths = [self.mask_paths[k] for k in indexes]
        X, y = self.__data_generation(image_paths, mask_paths)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    # https://lindevs.com/apply-gamma-correction-to-an-image-using-opencv
    def gammaCorrection(self, src, gamma):
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv2.LUT(src, table)

    """Correct normalization is important.."""

    def normalize_array(self, img):
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val)
        return normalized_img

    def __data_generation(self, image_paths, mask_paths):
        # Load images and masks
        X = np.empty((self.batch_size, *self.image_size, self.img_channels),  dtype=np.float32)
        y = np.empty((self.batch_size, *self.image_size, self.mask_channels))

        for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            img = plt.imread(image_path)
            img = cv2.resize(img, self.image_size)
            img = self.normalize_array(img)
            X[i,] = img

            mask = plt.imread(mask_path)
            mask = mask[:, :, :3] # Omit Alpha channel.. 

            # Set background pixels to 0 and object pixels to 1
            mask[mask == 0] = 1
            mask[mask == 255] = 0
            # print(mask.shape)
            mask = np.mean(mask, axis=2)
            # print(mask.shape)

            mask = cv2.resize(mask, self.image_size)
            mask = np.reshape(mask, (224, 224, 1))
            mask = mask.astype('int') 
            #print("mask unique3 :", np.unique(mask))
            #mask = mask.astype(np.uint8)
            y[i,] = mask
        

         # Data augmentation
        if self.augmentation:
          seed = np.random.randint(1, 100)

          data_gen_args = dict(horizontal_flip=True,
                              vertical_flip=True,
                               )
          
          image_data_generator = ImageDataGenerator(**data_gen_args)
          mask_data_generator = ImageDataGenerator(**data_gen_args)

          image_data_generator.fit(X, augment=True, seed=seed)
          mask_data_generator.fit(y, augment=True, seed=seed)

          X = image_data_generator.flow(X, batch_size=self.batch_size, shuffle=False, seed=seed)
          y = mask_data_generator.flow(y, batch_size=self.batch_size, shuffle=False, seed=seed)
          
          X, y = next(zip(X, y))

        y = y.astype(int)
        X = np.round(X,3)  #np.round(x,4) or X.astype(np.float32)
        return X, y

# Training generator
training_generator = DataGenerator(train_image_paths, train_mask_paths, shuffle=True, augmentation=True)

# Validation generator
validation_generator = DataGenerator(test_image_paths, test_mask_paths, shuffle=True, augmentation=True)
```
**Data augmentation**

Data augmentation is a technique used in deep learning to artificially increase the size of the training dataset by creating new training samples through transformations of the original data. For example, if you are working with images, you can apply various image transformation techniques like cropping, flipping, rotating, zooming, or adding noise to create new images from the original dataset. It helps to improve generalization, mitigate overfitting issues, and reduce the need for more data. 

In the subsequent steps, we begin by loading **ResNet50** as the base model and set the weights to 'imagenet'. To accomplish our task, we trim ResNet50 by obtaining the output of the fourth convolutional block. We then proceed to finetune the model by freezing the layers up to 'conv4_block4_2_conv', while the remaining layers will be trained.

```Python
# Load ResNet50 as the base model
model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')

# Get the output of the fourth convolutional block
conv4_block6_out = model.get_layer('conv4_block6_out').output

# Create a new model that outputs the feature maps of the fourth convolutional block
base_model = tf.keras.models.Model(inputs=model.input, outputs=conv4_block6_out)
base_model.trainable = True
set_trainable = False

# Freeze the layers of the base model and fine tune. 
for layer in base_model.layers:
    if layer.name == 'conv4_block4_2_conv':
      set_trainable == True
    if set_trainable:
      layer.trainable = True
    else:
      layer.trainable = False

base_model.summary()
```

We have determined the specific layers from ResNet50 to be utilized during the deconvolution process. These layers can be concatenated during this step.

```Python
conv3_block3_out = base_model.get_layer('conv3_block3_2_conv').output # conv3_block3_out
conv2_block3_out = base_model.get_layer('conv2_block3_3_conv').output # conv2_block3_out
conv1_conv = base_model.get_layer('conv1_conv').output
```

In the U-Net architecture, a bottleneck layer is introduced between the contracting and expanding paths. This bottleneck layer is essentially the bottleneck of the network where the number of feature maps is the smallest. It serves the purpose of bridging the contracting path to the expanding path while maintaining the high-resolution features.

```Python
# Bottle neck
bn = base_model.get_layer('conv4_block6_3_conv').output # conv4_block6_out
```
Then up convolution is defined as follow. Here we also use Batch Normalization. The primary purpose of batch normalization is to address the internal covariate shift problem. Internal covariate shift occurs when the distribution of the input to a layer changes as the parameters of the previous layers are updated during training. This makes it difficult for the subsequent layers to learn and adapt to the new distribution of inputs.

```Python
# Apply up-convolution
up1 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn)
up1 = tf.keras.layers.concatenate([up1, conv3_block3_out], axis=-1)
conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(up1)
conv5 = tf.keras.layers.Dropout(0.25)(conv5)
conv5 = tf.keras.layers.BatchNormalization()(conv5)
conv5 = tf.keras.layers.ReLU()(conv5)

up2 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
up2 = tf.keras.layers.concatenate([up2, conv2_block3_out], axis=-1)
conv6 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(up2)
conv6 = tf.keras.layers.Dropout(0.25)(conv6)
conv6 = tf.keras.layers.BatchNormalization()(conv6)
conv6 = tf.keras.layers.ReLU()(conv6)

up3 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
up3 = tf.keras.layers.concatenate([up3, conv1_conv], axis=-1)
conv7 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(up3)
conv7 = tf.keras.layers.Dropout(0.25)(conv7)
conv7 = tf.keras.layers.BatchNormalization()(conv7)
conv7 = tf.keras.layers.ReLU()(conv7)

conv7 = UpSampling2D(size=(2, 2))(conv7)
# Add a convolutional layer to reduce the number of channels to 3
output = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
model.summary()
```
Both dice coefficient and intersection over union (IoU) are evaluation metrics used in image segmentation tasks to measure the similarity between two sets of pixels. Dice coefficient measures the overlap between two masks, while IoU measures the intersection over the union of the two masks. Both metrics range from 0 to 1, with 1 indicating perfect overlap and 0 indicating no overlap.

```Python
def dice_coeff(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1):
    dc = dice_coeff(y_true, y_pred, smooth=smooth)
    loss = 1.0 - dc
    return loss

def iou(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)
```
Now time to compile the model. The purpose of setting clipnorm to a value is to prevent the gradients from becoming too large during training (sometimes we ended up with **Nan** value), which can cause the optimization algorithm to overshoot the minimum of the loss function and lead to poor convergence or divergence.
```Python
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
model.compile(optimizer = optimizer, loss= dice_loss, 
              metrics = ['accuracy',dice_coeff, iou])
checkpoint_filepath = '/path_to_the_checpoint_file/'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    save_best_only = True)

earlystop = EarlyStopping(patience=7, verbose=1)
history = model.fit(training_generator,epochs=20,
                    validation_data = validation_generator,
                    use_multiprocessing=True,callbacks=[checkpoint, earlystop],
                    workers=6, verbose=1)
```
Following figure shows training and validation loss after 20 epochs.
{{< figure class="center" src="/images/lossfun.png" >}}

Now let us save the model and test for unseen data;
```Python
model.save("./model.h5")

# Load the saved model
model_path = "./model.h5"

custom_objects = {'dice_loss':dice_loss, 'dice_coeff':dice_coeff, 'iou': iou}
modelhand = load_model(model_path, custom_objects=custom_objects)

# Load the saved weights
modelhand.load_weights(model_path)
```
Following are a few predicted segmentations for unseen data.
{{< figure class="center" src="/images/unseen.png" >}}

**Conclusion**

We have attempted to create a U-Net by utilizing the ResNet50 architecture along with its pre-trained 'imagenet' weights. This approach is particularly useful since we can leverage the benefits of pre-trained networks to achieve better performance and efficiency in our model.


