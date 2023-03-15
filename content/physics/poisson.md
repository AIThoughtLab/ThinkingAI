+++
author = "Mohamed ABDUL GAFOOR"
date = "2023-01-25"
title = "Solution of Poisson's equation over a polygonal domain"
slug = "poisson"
tags = [
    "Physics Based Learning",
    "PDE"
]
categories = [
    "Artificial Intelligence"
]

+++

**Poisson's Equation**

*----------------------------------------------------------------------------------------------------------------------------------------------*
{{< figure class="center" src="/images/Poisson.png" >}}
*----------------------------------------------------------------------------------------------------------------------------------------------*

Poisson's equation is a fundamental partial differential equation that arises in many areas of science and engineering. It is used to describe the behavior of scalar fields in physical systems such as electric potential or temperature distribution. The equation is expressed in terms of a source term and describes the distribution of the scalar field in a given physical system. Solving Poisson's equation is a fundamental problem in many areas of research, including mathematical physics, engineering, and applied mathematics. The solution of Poisson's equation can provide insight into the behavior of complex physical systems, such as electromagnetic fields, heat transfer, and fluid dynamics. 

In this post, we will delve into the topic of Poisson's equation over a polygonal domain and explore how to solve them using a deep learning technique. Specially, we will employ a Python library called "DeepXDE" to accomplish this task. Our polygonal domain has the following shape;

We will solve the following equation over a rectangle domain \\(\Omega = [0, 3] \times  [0, 2]\\);

$$\nabla^2u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = -2y $$

with the Dirichlet boundary conditions of \\( u(0,0) = 0, u(3, 0) = 0, u(3, 2) = 18, u(0, 2) = 0\\). The exact solution to the above equation is \\(u(x, y) = x^2y\\)

{{< figure class="center" src="/images/cg1.png" >}}

Our computational domain coordinates are \\([0, 0], [3, 0], [3, 2], [0, 2].\\)

```Python
import deepxde as dde
import numpy as np
import tensorflow as tf

geom = dde.geometry.Rectangle([0,0], [3,2])
```

**deepxde.geometry.geometry_2d.Rectangle(xmin, xmax)** takes two argments, xmin – coordinate of bottom left corner & xmax – coordinate of top right corner.

Next, we define the residual of the Poisson equation;

```Python
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 2*y
```

In this code, i and j are arguments to the **dde.grad.hessian** function, which computes the Hessian matrix of y with respect to x and y. The Hessian matrix is a matrix of second-order partial derivatives of a scalar-valued function. In two dimensions, the Hessian matrix has the following form:

```
[ d²u/dx²  d²u/dxdy ]
[ d²u/dydx   d²u/dy²]
```
Let us define our boundary conditions; 

```Python
def boundary(x, on_boundary):
    return on_boundary
```
    
In the above method, the **on_boundary** argument is a Boolean variable that indicates whether a given point in the computational domain is located on the boundary or not. The function returns the value of on_boundary, which is True if the point is on the boundary and False otherwise.

Dirichlet boundary conditions are defined here as; 
```
bc = [
    dde.DirichletBC(geom, lambda x: 0, boundary),
    dde.DirichletBC(geom, lambda x: 0, boundary),
    dde.DirichletBC(geom, lambda x: 18.0 if (np.isclose(x[0].any(), 3.0) and np.isclose(x[1].any(), 2.0)) else 0.0, boundary),
    dde.DirichletBC(geom, lambda x: 0, boundary)
]
```

Now we will define the exact solution to our problem;

```Python
# Exact solution to the PDE
def func(x):
  return x[:, 0:1]**2 * x[:, 1:]
```

We can ignor the above method, if we do not know the exact solution to the problem. Let us generate the data now;
```Python
data = dde.data.PDE(geom, pde, bc, num_domain=4000, solution=func, num_boundary=2000, train_distribution='Hammersley', num_test=5000)
```
The **data** we obtained above have several attributes and methods that we can access;

```Python
print(dir(data))
>> ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', 'add_anchors', 'anchors', 'auxiliary_var_fn', 'bc_points', 'bcs', 'exclusions', 'geom', 'losses', 'losses_test', 'losses_train', 'num_bcs', 'num_boundary', 'num_domain', 'num_test', 'pde', 'replace_with_anchors', 'resample_train_points', 'soln', 'test', 'test_aux_vars', 'test_points', 'test_x', 'test_y', 'train_aux_vars', 'train_distribution', 'train_next_batch', 'train_points', 'train_x', 'train_x_all', 'train_x_bc', 'train_y']

```
It is equally important to ensure the following in order to avoid any errors:

```Python
print(type(data.train_x))
print(type(data.train_y))

>>> <class 'numpy.ndarray'>
>>> <class 'numpy.ndarray'>
```

**Network**

Next we define a fully connected neural network (FNN) with 2 input nodes, 4 hidden layers of 50 nodes each with a tanh activation function, and 1 output node. The weights are initialized using Glorot uniform distribution and a dropout rate of 0.25 is applied to the hidden layers during training.

```Python
net = dde.nn.FNN([2] + [50] + [50] + [50] + [50] + [1], "tanh" , "Glorot uniform", dropout_rate= 0.25)

model = dde.Model(data, net)
model.compile("adam", lr=0.0001)

model.train(iterations=20000)

model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
```

The code above calls **model.compile** twice because the first time it sets up the training process with the Adam optimizer and a learning rate of 0.0001. The second call only sets the optimizer argument to "L-BFGS", which is another optimization algorithm used for training neural networks.

The goal of using different optimization algorithms is to find the best parameters for the model that minimize the loss function. Adam is a stochastic gradient descent method that uses adaptive learning rates, while L-BFGS is a quasi-Newton method that approximates the Hessian matrix of the loss function. Both have their own strengths and weaknesses, and it is common to try different optimization algorithms to find the best one for a particular problem.

After calling **model.train** with 20000 iterations, the loss history and training state are saved using **dde.saveplot**. Following figure shows the training and testing loss for different steps;

{{< figure class="center" src="/images/loss1.png" >}}

Now let us visualize the predicted solution and exact solution to the given Poisson equation. Following code shows the output from the exact solution formula;
```Python
def func(x):
  return x[:, 0:1]**2 * x[:, 1:]
  
y_true = func(train_state.X_test)
```

The figure below shows the solution generated using PINN and the exact solution;
{{< figure class="center" src="/images/solutions.png" >}}


**Conclusion**

To sum up, Physics-Informed Neural Networks (PINNs) provide an effective solution to ODE or PDE problems by leveraging both physics-based modeling and deep learning. They are capable of handling complicated geometries and boundary conditions, have low data requirements when compared to traditional numerical methods, and can rapidly produce precise results across a broad range of issues. Moreover, since PINNs can include physical laws and restrictions into the model, the results they produce are more interpretable and have greater physical meaning.

**Reference:**
1. https://deepxde.readthedocs.io/en/latest/index.html


