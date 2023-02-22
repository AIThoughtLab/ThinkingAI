+++
author = "Mohamed ABDUL GAFOOR"
date = "2019-04-20"
title = "Gradient Decent, Stochastic Gradient Descent and Adam Optimization Algorithm"
slug = "adam"
tags = [
    "Adam",
    "Machine Learning",
    "Deep Learning",
    "Optimizer"
]
categories = [
    "Artificial Intelligence"
]

+++

**Optimization Algorithms**

Before we start the Adaptive Moment Estimation (Adam) optimization algorithm, we first see what
is Gradient Decent (GD) optimization and how it works and then discuss about Stochastic Gradient
Descent (SGD) and finally Adam optimization. 

**Gradient Decent (GD)**

GD is one of the most well known optimization
algorithm available in machine learning and deep learning framework.
Lets say we have a scalar cost function C which is continuous in the domain \\(R_n\\) (n-dimensional real
vector space), takes the input vector \\(x ( = x_1, x_2 ... x_n)\\) of length \\(n\\) and we would like to find an
optimal value of this cost function. Assume our goal is to minimize the cost function \\(\forall x\\) such that such that
\\(C(x*) < C(x)\\) where \\(x*\\) is the n-dimensional vector that minimizes the cost function C. This
minimum cost value could be local minimum or global minimum but GD easily finds local
minimum if it is convex function. If it is non-convex, still there is a high chance the GD finds the
local minimum rather than global minimum. Hence the purpose of the GD is to travel from a
random initial point to a given local minimum x*.


Lets say the components of gradient of C is, \\(\triangledown C = [\partial C/\partial x_0, \partial C/\partial x_1, \partial C/\partial x_2 ... \partial C/\partial x_{n-1}]\\), where \\(\triangledown\\) is
the partial derivatives and \\(x_0, x_1 .. x_n\\) are components of vector \\(x\\). These partial derivatives
(\\(\partial C/\partial x_0, \partial C/\partial x_1\\)) tells how fast the directions are changing with respect to the basis vectors.
Imagine there is a unit vector \\(\hat{u}\\) and we are interested to project the gradient of C at a particular
point towards \\(\hat{u}\\), in other word, we are interested to find the **dot product** between the gradient and
the unit vector \\(\hat{u}\\). At a particular point \\(x\\), \\(\triangledown C(x)\\). \\(\hat{u} = |\triangledown C(x)||\hat{u}|cos\theta \\). Since \\(\hat{u}\\) is a unit vector, \\(|\hat{u}|\\) = 1.
Hence \\(|\triangledown C(x)| cos\theta = 1\\) is maximum when \\(cos\theta = 1\\). In other word, the direction of the gradient is exactly the direction of the unit vector (dot product is maximum when two vectors point in the same
direction). Similarly to minimize these two vectors must be in the opposite direction, this is our
interest, ie. to minimize the cost function.

GD Algorithm works as follow;
1. Start at \\(x_i\\)
2. Calculate the gradient at that point \\(i\\), \\(\triangledown C(x_{i})\\)
3. Go to the next point \\(x_{i+1}\\). i.e \\(x_{i+1} = x_{i} – \eta_{i} \triangledown C(x_{i})\\) for the minimization problem. Here \\\(\eta _{i}\\) is the step size.
4. Go to step 2 and calculate the new gradient at \\(x_{i+1}\\) until stopping criteria is met. When the
distance between \\(x_{i+1}\\) and \\(x_{i}\\) is less than some certain value, the algorithm stops and we assume \\(x_{i+1} ≈ x*\\). Hence \\(C(x*) < C(x)\\)

**Stochastic Gradient Descent (SGD)**

Now let’s move to Stochastic Gradient Descent (SGD). In SGD, we write down the cost function C
as finite sum of \\(C_k\\). i.e . \\(C_{x} = \sum_{k=1}^{k}C_{x}(x)\\). We can think of this as partial sum of a function. Then
we calculate the batch gradient, i.e sum of \\(\triangledown C_{x}(x)\\) for all \\(K[\sum_{k=1}^{k}\triangledown C_k(x)]\\), by substituting sum of
subset of partial gradient. Hence we introduce the stochastic gradient as \\(\triangledown \hat{C_k(x)} = \sum C_k(x)\\) for some
subset of k, where subset of k = 1, 2, 3 … K is chosen randomly. So in SGD, we follow exactly like
in GD except we replace gradient with Stochastic gradient (\\(\triangledown \hat{C_k}(x_i)\\)), then go to the 3rd step, i.e \\(x_{i+1} = x_{i} - \eta _{i}\triangledown \hat{C_k}(x_i)\\) for the minimization problem. Here \\(\eta _{i}\\) is the step size. The other steps are like in the GD to obtain \\(\hat{C(x*)} < C(x)\\).


**Adam optimization**

In Adam optimization, the gradient is updated from earlier steps using momentum method. The
word momentum comes from physics: the weight vector is very analogous to particles trajectories in
parameter space. In this case, we update the next step by linear combination of the current and the
previous update.

$$\Delta_i = x_i - x_{i-1}, x_{i} = x_{i-1} + \Delta_i$$
$$\Delta_i+1 = -\eta_i\triangledown \hat{C_k(x_{i})} + \beta_i\Delta_i -------(1)$$

The whole point to introduce momentum is for the faster convergence and to reduce the oscillation, so that it does not stuck in the local minima.

Like Diederik P.K et al. described in the original paper of Adam, we define the algorithm as follow. In addition to having decaying squared gradients vt like Adadelta and RMSprop have, in Adam we also store decaying average of previous gradients \\(m_t\\). We can compute decaying gradients and square gradients \\(m_t\\) and \\(v_t\\) as follow;

Like _Diederik P.K et al_. described in the original paper of Adam, we define the algorithm as follow.
In addition to having decaying **squared gradients** \\(v_t\\) like Adadelta and RMSprop have, in Adam we
also store decaying average of previous gradients \\(m_t\\). We can compute decaying gradients and
square gradients \\(m_t\\) and \\(v_t\\) as follow;

$$m_t = \beta_1m_{t-1} + (1-\beta_1)\triangledown_t$$

$$v_t = \beta_2v_{t-1} + (1-\beta_2)\triangledown^2_t$$

Since \\(m_t\\) & \\(v_t\\) are initialized to 0 vectors, they discovered these quantities are biased towards zero
during the initial time step or if \\(\beta_1\\), \\(\beta_2\\) are closer to one. To overcome this issue, bias-corrected  \\(m_t\\) & \\(v_t\\) have been introduced.

$$\hat{m_t} = \frac{m_t}{1-\beta_{1}^{t}}$$
$$\hat{v_t} = \frac{v_t}{1-\beta_{2}^{t}}$$

Hence,
$$\hat{m_t} = \frac{\beta_1m_{t-1} + (1-\beta_1)\triangledown_t}{1-\beta_{1}^{t}}$$
$$\hat{v_t} = \frac{\beta_2v_{t-1} + (1-\beta_2)\triangledown^2_t}{1-\beta_{2}^{t}}$$

Now if we use this to update our input vector \\(x ( = x_1, x_2 ... x_n)\\), which is our parameter and \\(\epsilon = 10^{-8}\\)
$$x_{t+1} = \frac{x_t - \hat{m_t}}{\hat{m_t} + \sqrt{\hat{v_t}} + \epsilon}$$

and then finally we add the corrected bias to the equation above.

_Diederik P.K et al_. in their paper suggest default values of 0.9 for \\(\beta_1\\), 0.999 for \\(\beta_2\\). And they showed
Adam optimization work well compare to the other optimization methods. 

The Algorithm works as follow,

1. \\(\eta\\) : step size
2. \\(\beta_1\\) and \\(\beta_2\\) \\(\epsilon \\) [0, 1) Exponential decay rates for the moment estimates.
3. \\(C(x)\\): Stochastic objective function with parameter \\(x\\)
4. \\(x_0\\): Initial parameter vector

	* \\(m_0\\) ← 0 Initialize the momentum vector (previous gradients: 1st momentum vector)
	* \\(v_0\\) ← 0 Initialize the squared gradients (2nd momentum vector)
	* t ← 0 Initialize the time steps
	
	  **while** \\(x_t\\) not converged do
	  
	  t ← t + 1 update the time
	  
	  \\(\triangledown_t\\) ← \\(\triangledown \hat{C_t} (x_{t-1})\\) Gradients w.r.t. stochastic objective \\(\hat{C}\\) at time-step t. 
	  
	  \\(m_t\\) ← \\(\beta_1 m_{t-1}\\) + \\((1-\beta_1)\triangledown_t\\) update without bias-corrected 1st momentum.
	  
	  \\(v_t\\) ← \\(\beta_2 v_{t-1}\\) + \\((1-\beta_2)\triangledown^{2}_t\\) update without bias-corrected 2nd momentum.
	  
	  \\(\hat{m}_t\\) ← \\(m_t\\) /(\\(1-\beta^t_1\\)) compute the bias-corrected 1st momentum.
	  
	  \\(\hat{v}_t\\) ← \\(v_t\\) /(\\(1-\beta^t_2\\)) compute the bias-corrected 2nd momentum.
	  
	  \\(x_t\\) ← \\(x_{t-1}\\) - \\(\frac{\hat{m_t}}{\hat{m_t} + \sqrt{\hat{v_t}} + \epsilon}\\) update the parameter.
	  
	  **end while**
	  
	  return \\(x_t\\)


Reference:
1. https://www.textbook.ds100.org/ch/11/gradient_stochastic.html
2. https://en.wikipedia.org/wiki/Stochastic_gradient_descent
3. http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentoptimizationalgorithms
4. https://www.quora.com/What-is-an-intuitive-explanation-of-the-Adam-deep-learning-optimization-algorithm
5. https://www.quora.com/Can-you-explain-basic-intuition-behind-ADAM-a-method-for-stochastic-optimization
6. https://arxiv.org/pdf/1412.6980v8.pdf






