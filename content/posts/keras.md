+++
author = "Mohamed ABDUL GAFOOR"
date = "2019-04-10"
title = "Keras – High Level API"
slug = "kerasAPI"
tags = [
    "Keras",
    "Machine Learning",
    "Deep Learning",
    "notMNIST"
]
categories = [
    "Artificial Intelligence"
]

+++

**What is Keras?**

It is a high-level neural network API, written in Python and able to running on top of TensorFlow. It is a very useful API, which enables us fast experimentation. Keras acts as an interface for the TensorFlow library. The task here is to study the image classification problem using notMNIST dataset. This dataset contains images of letters from A – J inclusive ([dowload here](https://drive.google.com/file/d/1QpI1AWvSAn_B61FPpJFPFgmHQ-5s05df/view?usp=share_link)). See the figure below to get an idea;

{{< figure class="center" src="/images/notMNIST.png" >}}

There are few pre-processing work has been completed already for this data and on top of it, it has been normalized. In total there are 200,000 training
images and 17,000 test images in this dataset. Image size is 28*28 pixels, so there are 784 features in total.

Now we will use Keras to build a SoftMax classifier. This will serve as a benchmark for the following parts. 

Import the necessary libraries first. 

```Python
from keras import models
from keras import layers
from keras.optimizers import sgd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pylab 
%matplotlib inline
import time
import tensorflow as tf
tf.reset_default_graph()
import h5py
import numpy as np
```
We will write a small function to load the data;
```Python
def loadData():
    with h5py.File('data.h5','r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        allTrain = hf.get('trainData')
        allTest = hf.get('testData')
        npTrain = np.array(allTrain)
        npTest = np.array(allTest)
        print('Shape of the array dataset_1: \n', npTrain.shape)
        print('Shape of the array dataset_2: \n', npTest.shape)
    return npTrain[:,:-1], npTrain[:, -1], npTest[:,:-1], npTest[:, -1]

#Loaded.  
x_train, y_train, x_test, y_test = loadData()   
```

We will initilize the model using keras **models.Sequential()**. There are 10 classes and input shape is at the begining **(784, )** because there are 28*28 pixels. We also set the activation function to **softmax**, optimizer to **adam**. We will take the advantage of batch size, hence set to 256.

```Python
model = models.Sequential()
model.add(layers.Dense(10, activation='softmax', input_shape=(784,)))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=30, batch_size = 256, validation_split=0.1)
```
After we train the model for 30 epochs, we will plot the graph to see the nature of the loss.
```Python
history_dict = history.history
print("Keys: ", history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(history_dict['acc'])+1)

def plot_results(loss_values,val_loss_values, epochs, **kwargs):
  label1 = kwargs.pop('label1')
  plt.plot(epochs, loss_values, 'bo',label = label1)
  label2 = kwargs.pop('label2')
  plt.plot(epochs, val_loss_values, 'r',label = label2)
  title  = kwargs.pop('title')
  xlabel = kwargs.pop('xlabel')
  ylabel = kwargs.pop('ylabel')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()

#loss  
plot_para = {'title': 'Training and Validation Loss', 'xlabel':'Epochs', 'ylabel': 'Loss', 'label1': 'Training Loss', 'label2': 'Validation Loss'}
plot_results(loss_values,val_loss_values, epochs, **plot_para) 

#accuray
plot_para = {'title': 'Training and Validation Accuracy', 'xlabel':'Epochs', 'ylabel': 'acc', 'label1': 'Training acc', 'label2': 'Validation acc'}
plot_results(acc_values,val_acc_values, epochs, **plot_para) 
```
Following figure below shows the loss and the accuracy for each epochs. We should be able to get an accuracy of 85% in the training data-set and around 83% in the test dataset. 
{{< figure class="center" src="/images/1.png" >}}

Now let us create a 2 layer networks. Layer 1 consists of 200 Neurons (activation='relu'), Layer 2 has 10 neurons (activation='softmax'). This time we can also test other optimizers, such **sgd**.

```Python
model = models.Sequential()
model.add(layers.Dense(200, activation='relu', input_shape=(784,)))
model.add(layers.Dense(10, activation='softmax'))
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)         
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer=sgd)
start_time = time.time()
history = model.fit(x_train, y_train, epochs=30, batch_size = 256, validation_split=0.1)
duration = time.time() - start_time
print("Total time is: ", round(duration, 2), "seconds")
```
This simple modification help us to improve the performance of the network significantly. For the same batch size like before, after 30 epochs, we should be able to get an accuracy of 90%.

```Python
history_dict = history.history
print("Keys: ", history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(history_dict['acc'])+1)

def plot_results(loss_values,val_loss_values, epochs, **kwargs):
  label1 = kwargs.pop('label1')
  plt.plot(epochs, loss_values, 'bo',label = label1)
  label2 = kwargs.pop('label2')
  plt.plot(epochs, val_loss_values, 'r',label = label2)
  title  = kwargs.pop('title')
  xlabel = kwargs.pop('xlabel')
  ylabel = kwargs.pop('ylabel')
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()
  
plot_para = {'title': 'Training and Validation Loss', 'xlabel':'Epochs', 'ylabel': 'Loss', 'label1': 'Training Loss', 'label2': 'Validation Loss'}
plot_results(loss_values,val_loss_values, epochs, **plot_para) 
```
Following figure shows the accuracy improvemet.
{{< figure class="center" src="/images/2.png" >}}

Now let us investigate what will happen, if we increase the depth of the neural network. For example we will try in the layer-1 400 neurons, in the layer-2 200 neurons and finally the layer-3 is softmax. 
```Python
model = models.Sequential()
model.add(layers.Dense(400, activation='relu', input_shape=(784,)))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)         
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer=sgd)
start_time = time.time()
history = model.fit(x_train, y_train, epochs=20, batch_size = 256, validation_split=0.1)
duration = time.time() - start_time
print("Total time is: ", round(duration, 2), "seconds")
```
We can observe a slight over-fitting in the case.
{{< figure class="center" src="/images/3.png" >}}

**Overfitting**

In ML/DL, the overfitting means where a model has been trained too well on the training data, so that it even captures the noises or random fluctuations in the data, instead of the underlying pattern. Because of this poor generalization and the trained model may not perform well on unseen data. Overfitting can occur when a model has too many parameters relative to the number of training examples, or when a model is too complex for the given problem. There are few ways we can handle this situation; such as early stopping, regularization, dropout or cross-validation. Following table shows how to spot over and underfitting during the training process. 

{{< figure class="center" src="/images/fit.png" >}}

Let us increase the model complexity even further. May be 4 layer networks configuration this time. In the lasyer-1 600 neurons; layer-2 400 neurons, layer-3 200 neurons and the layer-4 is softmax. 
Layer 1- 600 Neurons, Layer 2- 400 Neurons, Layer 3- 200 Neurons , Layer 4 – Softmax.

```Python
model = models.Sequential()
model.add(layers.Dense(600, activation='relu', input_shape=(784,)))
model.add(layers.Dense(400, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)         
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(loss='mean_squared_error', optimizer=sgd)
start_time = time.time()
history = model.fit(x_train, y_train, epochs=20, batch_size = 256, validation_split=0.1)
duration = time.time() - start_time
print("Total time is: ", round(duration, 2), "seconds")
```
We can clearly see now, the overfitting increases drastically with the number of layer and neurons. This is an important hyperparameter to workaround. 

{{< figure class="center" src="/images/overfit.png" >}}

If we test different regularization techniques, such as L1, L2 and dropout, it is clear that the L1 regularization perfomance is quite bad and L2 performence is very well. 
{{< figure class="center" src="/images/regularization.png" >}}

Dropout of **0.2** is performing well with accuracy of 93% on the test data compared to the **0.5** dropout.

