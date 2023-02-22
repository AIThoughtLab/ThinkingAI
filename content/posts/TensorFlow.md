+++
author = "Mohamed ABDUL GAFOOR"
date = "2019-04-01"
title = "TensorFlow and the Low Level API - Part 1"
slug = "Fashion-MNIST_TF"
tags = [
    "TensorFlow",
    "Deep Learning",
    "MNIST"
]
categories = [
    "Artificial Intelligence"
]

+++

**What is TensorFlow?**

TensorFlow is an open-source software library for dataflow and differentiable programming across a range of tasks. It is primarily used for machine learning and deep learning applications. TensorFlow provides a high-level API for building and training machine learning models, as well as a low-level API for defining mathematical operations. With TensorFlow, users can easily build, train, and deploy complex models on a variety of platforms, including desktops, servers, and mobile devices. Additionally, TensorFlow has a large community of developers and users, which makes it a popular choice for machine learning projects. We will use TensorFlow to develop a machine learning model using Fashion-MNIST dataset. The Fashion-MNIST is an image dataset consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image (784 pixel values in total), associated with a label from 10 different classes. The following are the set of classes in this classification problem (the associated integer class label is listed in brackets). Followings are categories;

* T-shirt/top (0)
* Trouser (1)
* Pullover (2)
* Dress (3)
* Coat (4)
* Sandal (5)
* Shirt (6)
* Sneaker (7)
* Bag (8)
* Ankle boot (9)

For example, you can load the as follow;

```Python
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

Let us build a binary two layer classifier using the TensorFlow low level API. So we will extract two classes from the training and test data (map the associated labels to 0 and 1).
For instance, let us say **dress** and **bag**. The integer class label for a dress is 3 and the integer class label for a bag is 8. We will extract all data related to just these two classes from the
training and test data. Hence, our class labels for the train and test data will have the integer values 3 or 8. Now we will map all labels encoded as 3 to 0 and all labels encoded as 8 to 1. The reason for this is that the Sigmoid function is a binary classifier that can only outputs values between 0 and 1.

```Python
# Extract only 3 and 8 from train dataset
train_labels_new = train_labels[(train_labels == 3) | (train_labels == 8)]
train_images_new = train_images[(train_labels == 3) | (train_labels == 8)]

# Extract only 3 and 8 from test dataset
test_labels_new = test_labels[(test_labels == 3) | (test_labels == 8)]
test_images_new = test_images[(test_labels == 3) | (test_labels == 8)]


# Reshape training dataset so that the features are flattened
train_images_new = train_images_new.reshape(train_images_new.shape[0], -1).astype('float32')
test_images_new = test_images_new.reshape(test_images_new.shape[0], -1).astype('float32')

#Normalized the data
train_images_new = train_images_new / 255.0
test_images_new = test_images_new / 255.0


# encoding 3=0 and 8=1
test_labels_new[test_labels_new == 3] = 0
test_labels_new[test_labels_new == 8] = 1

train_labels_new[train_labels_new == 3] = 0
train_labels_new[train_labels_new == 8] = 1

#Reshape the label dataset
test_labels_new = test_labels_new.reshape(1,-1)
train_labels_new = train_labels_new.reshape(1,-1)

#Transpose the train and test data
train_images_new = train_images_new.T
test_images_new = test_images_new.T
```
Now let us define the learning rate. The learning rate in Machine Learning is a hyperparameter that determines the step size at which the optimizer makes updates to the model parameters. It plays a vital role in training a model because it controls the speed and direction of the updates. If the learning rate is quite high, the model's parameters will be updated too quickly, causing the optimization to overshoot the minimum and converge slowly or even oscillate and never converge. On the other hand, if the learning rate is very low, the model's parameters will be updated too slowly, causing the optimization to converge slowly.

Setting an appropriate learning rate can make a big difference in the model's performance and the training time. In general, the learning rate should be set such that it strikes a balance between convergence speed and model performance. Finding an optimal learning rate is often an iterative process that involves experimenting with different values and observing their impact on the training process and the final model performance.

```Python
learningRate = 0.01
# Define the placeholders to load our training and target labels.
x = tf.placeholder(tf.float32, [train_images_new.shape[0], None])
y_ = tf.placeholder(tf.float32, [1, None])
```
Also we will define our weight and bias matrix. Set Layer 1: 100 neurons (ReLu activation function) and Layer 2: 1 neuron (Sigmoid activation function).

```Python
# define our weight and bias matrix
w = tf.Variable(tf.random_normal([train_images_new.shape[0],100], mean=0.0, stddev=0.8))
w_T = tf.transpose(w)
b = tf.Variable([0.])

#Multiply weight and bias matrix
y_pred_1 = tf.matmul(w_T, x) + b
#pipe it through relu activation function
layer1 = tf.nn.relu(y_pred_1)

# define our weight and bias matrix
w1 = tf.Variable(tf.random_normal([100,1], mean=0.0, stddev=0.04))
w1_T = tf.transpose(w1)
b1 = tf.Variable([0.])
#Multiply weight and bias matrix
y_pred_2 = tf.matmul(w1_T, layer1) + b1
#pipe it through sigmoid activation function
y_pred_sigmoid = tf.sigmoid(y_pred_2)
```

We will use **tf.nn.sigmoid_cross_entropy_with_logits(logits=A2, labels=y)**. The pre-activation values, also known as logits, of a neuron are fed into the Sigmoid activation function. This function maps the logits to probabilities between 0 and 1, which can then be used to make a prediction. The predicted values are then compared with the actual labels (y) and the cross entropy error is calculated for each instance in the training data. The cross entropy error is a measure of how well the predicted values match the true labels and will be used to guide the optimization process in training the neural network.

```Python
# Get cross entropy error for all our training dataset
x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred_2, labels=y_)
# Get the mean cross entropy error
loss = tf.reduce_mean(x_entropy)
# Apply Gradient Descent to minimize the error.
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
# Round off the predictions (0 or 1)
predictions = tf.round(y_pred_sigmoid)
predictions_correct = tf.cast(tf.equal(predictions, y_), tf.float32)
# Get the mean accuracy 
accuracy = tf.reduce_mean(predictions_correct)
```

Now time to start the session;

```Python
num_Iterations = 50
# Start the session
with tf.Session() as sess:
        # create lists to plot the graph
        train_loss_list = []
        epoch_list = []
        train_acc_list = []
        start_time = time.time()
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        for i in range(num_Iterations):
            _, train_loss, train_acc  = sess.run([optimizer, loss, accuracy], feed_dict={x:train_images_new, y_:train_labels_new})
            train_loss_list.append(train_loss)
            epoch_list.append(i)
            train_acc_list.append(train_acc*100)

            
            print ("Epoch ", i, " Train Loss: ", train_loss, "  Train Acc: ", train_acc)

        duration = time.time() - start_time

        pylab.plot(epoch_list, train_acc_list, '-ob', label = 'train_accuracy')
        pylab.plot(epoch_list, train_loss_list, 'purple', label = 'train_loss_list')
        pylab.legend(loc='bottom right')
        plt.xlabel('Epochs')
        pylab.show()
        print("Total time is: ", round(duration, 2), "seconds")
        print ("Final Test Accuracy ", sess.run(accuracy*100, feed_dict={x:test_images_new, y_:test_labels_new}))
```

You can easily get a 98% accuracy with the Fashion-MNIST dataset. 




