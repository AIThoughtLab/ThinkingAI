+++
author = "Mohamed ABDUL GAFOOR"
date = "2022-12-05"
title = "Second order Ordinary Differential Equation"
slug = "ode"
tags = [
    "Physics Based Learning",
    "ODE"
]
categories = [
    "Artificial Intelligence"
]

+++

**Second-Order Ordinary differential equation (ODE)**

*----------------------------------------------------------------------------------------------------------------------------------------------*

*--When a system is subject to a periodic forcing with a frequency that is equal to its natural frequency, it results in resonance. In forced oscillation, the frequency of the periodic force is denoted by \\(\omega\\) while the natural frequency of the system is denoted by \\(\omega_{0}\\). When \\(\omega\\) and \\(\omega_{0}\\) are equal, the amplitude of the oscillation increases significantly, leading to resonance. Resonance can be observed in a variety of physical systems, including mechanical, electrical, and acoustic systems. It can be both useful and destructive, depending on the context in which it occurs.--*

*----------------------------------------------------------------------------------------------------------------------------------------------*

In this post, we will delve into the topic of second-order ordinary differential equations (ODEs) and explore how to solve them using a deep learning technique. Specially, we will employ a Python library called "DeepXDE" to accomplish this task.

A second-order ODE is a kind of differential equation that deals with a function's second derivative with respect to its independent variable. The standard form of a second-order ODE is;

$$y''(x) + p(x)y'(x) + q(x)y(x) = r(x)$$

where \\(y(x)\\) is the function that we are trying to find, and \\(p(x)\\), \\(q(x)\\), and \\(r(x)\\) are functions that are known to us.

This equation is applicable in various fields such as Physics, Engineering, and Mathematics, and is used to model different phenomena such as the movement of a mass on a spring, the vibrations of a string, and the flow of fluids.

Solving a second-order ODE involves finding a function \\(y(x)\\) that satisfies the equation for a specific set of initial or boundary conditions. This can be achieved through analytical methods such as separation of variables, integrating factors, and Laplace transforms, or numerical methods such as the Euler's method, the Runge-Kutta method, or finite difference methods.

In this post we will solve an ODE in the following form; 

$${y}'' + \omega _{0}^2y = \frac{F_0}{m} cos(\omega _{0}t)$$

with initial conditions;
$$ y(0) = 0,   y'(0) = \frac{F_0}{2m\omega _{0}} $$

Let us assume  \\(F_0 = 10N\\),  \\(m = 2kg\\),  \\(\omega _{0} = 4\\)..

Hence, our equation turns into;
$${y}'' + 4^2y = \frac{10}{2} cos(4t)$$

with the initial conditions;
$$ y(0) = 0,   y'(0) = 0.625 $$

The exact solution to this problem is;
$$y = \frac{5t.sin(4t)}{8} + 0.15625.sin(4t)$$

We can start the process by first defining a computational geometry for our simulation. One approach is to use a built-in class called TimeDomain. To do so, we can use the following code:
```Python
import deepxde as dde
import numpy as np

geom = dde.geometry.TimeDomain(0, 5) # Time interval between 0 to 5.
```

Next, we will define residual of the ODE, in the context of solving an ODE numerically, the residual refers to the difference between the left-hand side (LHS) and the right-hand side (RHS) of the ODE after substituting the numerical solution at a given time point.

```Python
def ode(t, y):
    d2y_dt2 = dde.grad.hessian(y, t)
    return d2y_dt2 + 16*y - 5*dde.backend.cos(4*t)
```

The first argument here is the t-coordinate and the second argument is the network output, i.e., the solution  \\(y(t)\\).

**Initial conditions**

Next, time to define initial conditions, when \\(t = 0; y(0) = 0\\).

```Python
# when t = 0; y(0) = 0
ic1 = dde.icbc.IC(geom, lambda x: 0, lambda _, on_initial: on_initial)
```

Similarly;  when \\(t = 0; y'(0) = 0.625\\). However, this is quite complicated as we deal with initial condition with a first derivative. 
We must write a function that should return True for those points satisfying t=0 and False otherwise. Note that because of rounding-off errors, it is often wise to use np.isclose to test whether two floating point values are equivalent [1].

Following is the function that returns the error of the initial condition, \\(y'(0)=0.625\\);

```Python
def boundary(t, on_boundary):
    return on_boundary and np.isclose(t[0], 0)

def error(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 0.625

# Second initial condition is
ic2 = dde.icbc.OperatorBC(geom, error, boundary)
```

**- There is a note regarding the use of the variable X in the func function. If X is used in func, then num_test should not be set when defining the PDE or TimePDE objects in DeepXDE. If num_test is set, then DeepXDE will throw an error.**


Now we will define the exact solution to our problem;

```Python
# Exact solution to the ODE
def func(t):
  return (5*t*np.sin(4*t))/8 + 0.15625*np.sin(4*t)
```

We can ignor the above method, if we do not know the exact solution to the problem. Let us generate the data now;
```Python
data = dde.data.TimePDE(geom, ode, [ic1, ic2], 160, 20, solution=func, num_test=500)

#  TimePDE has several attributes and methods that you can access
print(dir(data))
print(data.train_x.shape)
print(data.train_y.shape)

>>>
['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', 'add_anchors', 'anchors', 'auxiliary_var_fn', 'bc_points', 'bcs', 'exclusions', 'geom', 'losses', 'losses_test', 'losses_train', 'num_bcs', 'num_boundary', 'num_domain', 'num_initial', 'num_test', 'pde', 'replace_with_anchors', 'resample_train_points', 'soln', 'test', 'test_aux_vars', 'test_points', 'test_x', 'test_y', 'train_aux_vars', 'train_distribution', 'train_next_batch', 'train_points', 'train_x', 'train_x_all', 'train_x_bc', 'train_y']
(20, 1)
(20, 1)
```


The numbers 160 and 20 are used to control the number of training points that the neural network will use to learn the underlying pattern of the PDE. By increasing the number of training points, the neural network can learn a more accurate representation of the underlying pattern, which can lead to better predictions. However, increasing the number of training points also requires more computation time and resources. We use here 500 residual points for testing the ODE.


**Network**

Let us build our network. We build a fully connected network of with 3 hidden layers and each layer contains 60 neurons. 

```Python
layer_size = [1] + [60] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
```
Now that we have defined the ODE and the network architecture, we can proceed with building a Model object. To train the network, we first choose an optimizer, set the learning rate to 002, and then run the training loop for 15000 iterations. During training, we set the weight of the ODE loss to 0.01, and the weights of the two initial conditions to 0.1 and 1 respectively. We also compute the L2 relative error as a metric to evaluate the model's performance.

```Python
model = dde.Model(data, net)
model.compile("adam", lr=.002, loss_weights=[0.01, 0.1, 1], metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=15000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

```
Following figure shows the prediction and the exact solution. It is interesting to combine physics knowledge with machine learning. 
{{< figure class="center" src="/images/pinn1.png" >}}


**Conclusion**

In summary, Physics-Informed Neural Networks (PINNs) offer a powerful approach to solving ODE or PDE that combines the strengths of physics-based modeling and deep learning. PINNs can handle complex geometries and boundary conditions, require very little data compared to traditional numerical methods, and able to provide a fast and accurate solution to a wide range of problems. Additionally, PINNs can naturally incorporate physical laws and constraints into the model, leading to more interpretable and physically meaningful results.



**Reference:**

1. https://deepxde.readthedocs.io/en/latest/index.html



