+++
author = "Mohamed ABDUL GAFOOR"
date = "2019-04-03"
title = "TensorFlow and the Low Level API - Part 2"
slug = "Fashion-MNIST_TF2"
tags = [
    "TensorFlow",
    "Deep Learning",
    "MNIST"
]
categories = [
    "Artificial Intelligence"
]

+++

As a continuation from the previous part-1, in this post we will discuss a full **multi-class classification** problem (all 10 classes for Fashion MNIST). Using TensorFlow’s low level API (in graph mode) let us build a multi-layer neural network. We will define our architecture as follow:

* Layer 1: 300 neurons (ReLu activation functions).
* Layer 2: 100 neurons (ReLu activation function)
* Layer 3: Softmax Layer
* Learning rate: 0.01 (with Gradient Descent).

We will import the necessary libraries first;
```Python
import tensorflow as tf
import numpy as np
import pylab 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline
import time
tf.reset_default_graph()
```
Then we will write a small function to load the data, where we reshape, normalize and convert the labels to one-hot-encoding. 
**One-hot encoding** is a technique used in machine learning to represent categorical variables as numerical data. In one-hot encoding, a categorical variable with N possible values is transformed into N binary variables, with each binary variable representing one of the N possible values. Only one of the N binary variables is "hot" or "on", which means it is set to 1, and the rest of the variables are set to 0. This results in a binary vector of length N that uniquely represents the categorical variable.

For example, let us say there are A, B, and C, and one-hot encoding would transform this variable into three binary variables: [1, 0, 0], [0, 1, 0], [0, 0, 1]. The problem with this encoding technique is that, it can result in a large number of features and increase the dimensionality of the data, which can slow down the training process. To mitigate this issue, techniques such as dimensionality reduction and feature selection can be applied.


```Python
def loadData():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (x_test, y_test) = fashion_mnist.load_data()
    print(train_images.shape)
    x_train, x_valid, y_train, y_valid = train_test_split(train_images, train_labels, test_size=0.1, random_state=1)

    #reshape the data
    x_train = x_train.reshape(x_train.shape[0], -1).astype('float32')
    x_valid = x_valid.reshape(x_valid.shape[0], -1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')

    # Normalize
    x_train = x_train/255.0
    x_valid = x_valid/255.0
    x_test = x_test/255.0
    
    #Transpose of the matrix
    x_train = x_train.T
    x_valid = x_valid.T
    x_test = x_test.T

    # Convert labels to one-hot-encoded
    number_of_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, number_of_classes)
    y_test = tf.keras.utils.to_categorical(y_test, number_of_classes)
    print(y_train[0])
    print(y_train[1])
    
    # transpose the labels
    y_train = y_train.T
    y_valid = y_valid.T
    y_test = y_test.T

    print ("Reshaped Data: ")
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape)
    #print (train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    return x_train, y_train, x_valid, y_valid, x_test, y_test
```

Now we will build our model, where input vector is 784, in the first hidden layer there are 300 neurons and 100 in the second hidden layer. Finally the output layer has 10 neurons as there are 10 classes;

```Python
# Model building
def main(x_train, y_train, x_valid, y_valid, x_test, y_test):
    n_inputs = 784
    n_hidden_1 = 300
    n_hidden_2 = 100
    n_output = 10
    learningRate = 0.01
    n_epochs = 40
  
    # Placeholders for the traing and label data    
    x = tf.placeholder(tf.float32, [n_inputs, None], name='image')    
    y = tf.placeholder(tf.float32, [n_output, None], name='label')
   # tf.get_variable("W1", [n_hidden, n_inputs],initializer = tf.glorot_uniform_initializer(seed=1) )
    # Create weight and bias matrices (variables) for each layer of our network
    W1 = tf.Variable(tf.random_normal([n_hidden_1, n_inputs], mean=0.0, stddev= 0.9)) # 784 = 28 * 28
    b1 = tf.Variable(tf.zeros([n_hidden_1, 1]))

    W2 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], mean=0.0, stddev= 0.05)) # 784 = 28 * 28
    b2 = tf.Variable(tf.zeros([n_hidden_2, 1]))

    W3 = tf.Variable(tf.random_normal([n_output, n_hidden_2], mean=0.0, stddev=0.005)) # 784 = 28 * 28
    b3 = tf.Variable(tf.zeros([n_output, 1]))
  
    # Push feature data through layers of NN
    layer_1  = tf.nn.relu(tf.add(tf.matmul(W1, x), b1))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(W2, layer_1), b2))
    layer_3 = tf.add(tf.matmul(W3, layer_2), b3)

    soft_max = tf.nn.softmax(layer_3)
  
    err = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=layer_3)
    loss = tf.reduce_mean(err)
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(soft_max), tf.argmax(y))

   # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    with tf.Session() as sess:
        train_loss_list = []
        val_loss_list = []
        epoch_list = []
        #epoch_list1 = []
        train_acc_list = []
        val_acc_list = []
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(n_epochs):
            _, train_loss, train_acc  = sess.run([optimizer, loss, accuracy], feed_dict={x: x_train, y: y_train})
            _, val_loss, val_acc  = sess.run([optimizer, loss, accuracy], feed_dict={x: x_valid, y: y_valid})

            train_loss_list.append(train_loss)
            epoch_list.append(epoch)
            train_acc_list.append(train_acc)
            ##-------------------------------
            val_loss_list.append(val_loss)
            #epoch_list.append(epoch)
            val_acc_list.append(val_acc) 
            
            print ("Epoch ", epoch, " Train Loss: ", train_loss, "  Train Acc: ", train_acc)
            print ("Epoch ", epoch, " Val Loss: ", val_loss, "  Val Acc: ", val_acc)

        duration = time.time() - start_time

        pylab.plot(epoch_list, val_loss_list, '-or', label = 'val_loss')
        pylab.plot(epoch_list, train_acc_list, '-ob', label = 'train_accuracy')
        pylab.plot(epoch_list, val_acc_list, '-ok', label = 'val_accuracy')
        pylab.plot(epoch_list, train_loss_list, 'purple', label = 'train_loss_list')
        pylab.legend(loc='bottom right')
        plt.xlabel('Epochs')
        pylab.show()
        print("Total time is: ", round(duration, 2), "seconds")
        print ("Final Validation Accuracy ", sess.run(accuracy, feed_dict={x: x_valid, y: y_valid}))
        print ("Final Test Accuracy ", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

```
Following is the output.
{{< figure class="center" src="/images/keras.png" >}}

Hence the final validation accuracy is 0.6755 and the final test accuracy is 0.6767.

**Mini-batch**

Mini-batch training is a good approach in deep learning because it balances the trade-off between computational efficiency and stability of the gradients. It involves dividing the training dataset into small batches, which are processed in parallel to compute the gradients. This allows for faster training times as compared to batch training (where the entire dataset is processed at once), while still preserving the stability of gradients and avoiding the fluctuations seen in stochastic gradient descent (where only a single sample is processed at once). Mini-batch training also enables the use of parallel computing resources, such as GPUs, to speed up training. If we introduce a mini batch size of 100, it is indeed possible to increase the accuracy of the model. When we test, it was around 0.7287.  Introducing batch size is easy. Following is the code snippet.

```Python
def getBatch(x, y, begin, end):
    x_miniBatch = x[begin:end]
    y_miniBatch = y[begin:end]
    return x_miniBatch, y_miniBatch
```

There is no much change to the previous code except some modification should be done to the **tf.Session()**.

```Python
    with tf.Session() as sess:
        loss_list = []
        epoch_list = []
        acc_list = []
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        iteration = int(len(x_train)/batchSize)
        for epoch in range(n_epochs):
            #print('Training epoch: {}'.format(epoch + 1))
            for i in range(iteration):
                begin = i * batchSize
                end = (i + 1) * batchSize
                x_short, y_short = getBatch(x_train, y_train, begin, end)
                x_short = x_short.T
                y_short = y_short.T
                feed_dict = {x: x_short, y:y_short}
                sess.run(optimizer, feed_dict=feed_dict)
                
                if i % 100 == 0:
                    loss_after_batch, acc_after_batch = sess.run([loss, accuracy],feed_dict=feed_dict)
                    #print('iter: ', i, 'Loss: ', loss_after_batch, 'Training Accuracy: ', acc_after_batch)
            
            feed_dict_test = {x: x_test.T, y: y_test.T}
            loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
            #print('---------*************---------')
            #print("Test Loss: ", loss_test, "Test Accuracy: ", acc_test)
            return acc_test
```
Thats it!! Now we know how to use mini-batch in our ML/DL problems. 






