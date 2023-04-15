+++
author = "Mohamed ABDUL GAFOOR"
date = "2022-09-30"
title = "Understanding the Dempster-Shafer Theory for Deep Learning"
slug = "DS"
tags = [
    "Dempster-Shafer",
    "Machine Learning",
    "Deep Learning"
]
categories = [
    "Artificial Intelligence"
]

+++

**Introduction**

The Dempster-Shafer theory (DST), named after its creators Arthur Dempster and Glenn Shafer, is a powerful mathematical framework for representing and managing uncertain information. It is particularly useful in situations where the available data is incomplete, imprecise, or contradictory. In this post, we will introduce the core concepts of the Dempster-Shafer theory, explore its practical applications in deep learning, and discuss its advantages over traditional probability theory.


**Belief functions**

In the DST, uncertain information is represented by belief functions. A belief function, also known as a basic probability assignment (BPA), is a mapping from subsets of the set of possible outcomes, called the frame of discernment, to the interval [0, 1]. The sum of all BPAs over the power set of the frame of discernment (excluding the empty set) is equal to 1.

**Belief, plausibility, and interval-valued probability**

The belief function allows us to define two important measures: belief and plausibility. The belief of an event is the sum of the BPAs assigned to all subsets that contain the event. The plausibility of an event is the sum of the BPAs assigned to all subsets that intersect with the event. The interval between the belief and plausibility of an event is referred to as the interval-valued probability. In the following figure interval-valued probability is a representation of uncertainty.

{{< figure class="center" src="/images/ds1.png" >}}


**Dempster's rule of combination**

Dempster's rule of combination, which is a crucial aspect of the Dempster-Shafer theory, is a rule for combining two or more belief functions related to the same frame of discernment. The combined belief function is computed by normalizing the sum of the product of BPAs assigned to compatible subsets.

Given two belief functions \\(m1\\) and \\(m2\\), the combined belief function \\(m\\) is calculated for each subset A of the frame of discernment \\(\Theta\\) using the equation:

$$m(A) = \frac{\sum_{X \cap Y = A} m_{1}(X) \cdot m_{2}(Y)}{1 - K}$$

where the summation is over all pairs of subsets \\(X\\) and \\(Y\\) such that \\(X ∩ Y = A\\), and \\(K\\) is a normalization factor calculated as:

$$K = \sum_{X \cap Y = \emptyset} m_{1}(X) \cdot m_{2}(Y)$$

where the summation is over all pairs of subsets \\(X\\) and \\(Y\\) such that \\(X ∩ Y = ∅\\) (empty set).


COMPLICATED???

{{< figure class="center" src="/images/com.png" >}}

Let us see an example!! Suppose we want to diagnose a patient based on the symptoms they are experiencing. We have three possible diseases: Disease A \\((D_A)\\), Disease B \\((D_B)\\), and Disease C \\((D_C)\\). We have two different tests (Test 1 and Test 2) that provide evidence about the presence of these diseases.

Frame of discernment \\(\Theta\\):  We have two possible outcomes - Intruder Present \\(I\\) and Intruder Absent \\(A\\).
Frame of discernment \\(\Theta\\): \\({D_A, D_B, D_C}\\)

**Evidence from Test 1:**

Suppose Test 1 provides the following belief functions:

* Belief in Disease A \\((D_A)\\): 0.5
* Belief in Disease B \\((D_B)\\): 0.1
* Belief in Disease C \\((D_C)\\): 0.1
* Belief in Don't Know \\((D_A ∪ D_B ∪ D_C)\\): 0.3

**Evidence from Test 2:**

Suppose Test 2 provides the following belief functions:

* Belief in Disease A \\((D_A)\\): 0.3
* Belief in Disease B \\((D_B)\\): 0.4
* Belief in Disease C \\((D_C)\\): 0.1
* Belief in Don't Know \\((D_A ∪ D_B ∪ D_C)\\): 0.2

We can combine the belief functions from both tests using Dempster's rule. The combined belief function is computed by normalizing the sum of the product of BPAs assigned to compatible subsets.

To compute the combined belief for each hypothesis, we first calculate the joint mass;
\\(m(D_A) = m1(D_A) * m2(D_A) = 0.5 * 0.3 = 0.15\\)
\\(m(D_B) = m1(D_B) * m2(D_B) = 0.1 * 0.4 = 0.04\\)
\\(m(D_C) = m1(D_C) * m2(D_C) = 0.1 * 0.1 = 0.01\\)

Sum of all compatible subset product combinations;
```Python
m(D_A ∪ D_B ∪ D_C) = m1(D_A) * m2(D_A ∪ D_B ∪ D_C) + m1(D_B) * m2(D_A ∪ D_B ∪ D_C) + m1(D_C) * m2(D_A ∪ D_B ∪ D_C) + m1(D_A ∪ D_B ∪ D_C) * m2(D_A) + m1(D_A ∪ D_B ∪ D_C) * m2(D_B) + m1(D_A ∪ D_B ∪ D_C) * m2(D_C)
```

Hence, 

\\(m(D_A ∪ D_B ∪ D_C) = 0.5 * 0.2 + 0.1 * 0.3 + 0.1 * 0.4 + 0.1 * 0.2 + 0.1 * 0.1 + 0.3 * 0.3 = 0.1 + 0.03 + 0.04 + 0.02 + 0.01 + 0.09 = 0.29\\)

Now, we normalize the joint mass;

\\(K = 1 / (1 - m(D_A ∪ D_B ∪ D_C)) = 1 / (1 - 0.29) = 1 / 0.71 ≈ 1.408\\)

Finally, we calculate the normalized belief functions;

\\(Bel(D_A) = m(D_A) * K = 0.15 * 1.408 ≈ 0.211\\)
\\(Bel(D_B) = m(D_B) * K = 0.04 * 1.408 ≈ 0.056\\)
\\(Bel(D_C) = m(D_C) * K = 0.01 * 1.408 ≈ 0.014\\)

After combining the evidence from both tests, we have the following belief functions;

\\(Bel(D_A) ≈ 0.211\\),
\\(Bel(D_B) ≈ 0.056\\),
\\(Bel(D_C) ≈ 0.014\\)


Therefore, based on the combined belief values, Disease A appears to be the most likely diagnosis, followed by Disease B and Disease C. However, it is important to note that these belief values do not provide definitive answers but rather indicate the strength of the evidence for each disease hypothesis. Further tests or expert consultation may be necessary to make a more accurate diagnosis.

**Dempster-Shafer theory in deep learning**

Integrating the Dempster-Shafer theory into deep learning can help improve the robustness and interpretability of the models, especially in situations where the data is incomplete or ambiguous. Here are some steps to incorporate the Dempster-Shafer theory into our deep learning models;

* First of all, identify the problem that we want to solve using deep learning and define the frame of discernment \\(\Theta\\), which represents the set of all possible outcomes or classes.

* Design: Design a deep learning model, such as a convolutional neural network (CNN) to process the input data and generate the initial outputs or predictions. We can use standard deep learning frameworks like TensorFlow for this purpose.

* Generate belief functions: Instead of generating a single probability value for each class, modify the output layer of the deep learning model to generate belief functions (basic probability assignments or BPAs). These BPAs can be calculated based on the outputs from the softmax activation function, by transforming or mapping them according to the problem requirements and the level of uncertainty in our data.

* Combine belief functions: If we have multiple sources of evidence (e.g. motion sensor and a sound sensor) for the same problem, use Dempster's rule of combination to combine the belief functions obtained from these sources. This will result in a new belief function that provides a more comprehensive representation of the uncertainty in our data.

* Calculate belief and plausibility: Compute the belief and plausibility values for each class using the combined belief functions. These values provide an interval-valued probability, which can help us better understand the uncertainty associated with each class prediction.

* Model interpretation and decision-making: Use the belief and plausibility values to interpret the results of our deep learning model, and make more informed decisions based on these values. For example, we can choose the class with the highest plausibility value as the most plausible prediction or use the interval-valued probability to quantify the level of uncertainty in our model's predictions.

* Evaluate and fine-tune: Evaluate the performance of the Dempster-Shafer enhanced deep learning model using relevant metrics (e.g., accuracy, F1-score). Based on the evaluation results, fine-tune the model by adjusting its architecture or training parameters to achieve better performance.


**Conclusion**

In summary, incorporating the DST into deep learning can help us better handle uncertainty and improve the interpretability of our models, leading to more robust and reliable predictions in situations with incomplete or ambiguous data..











