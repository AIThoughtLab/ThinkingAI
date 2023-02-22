+++
author = "Mohamed ABDUL GAFOOR"
date = "2022-10-13"
title = "Bayesian optimization of time perception"
slug = "BayesianOptimization"
tags = [
    "Bayesian",
    "time perception",
    "Psychology"
]
categories = [
    "Bayesian Optimization"
]

+++


**Introduction: Context and Motivation**

Humans are accurate at timing sub-second to minute intervals in daily routines, but their experience of time can be biased in different contexts. Traditionally, these biases were explained by the decay of modality-specific representations, changes in the internal clock speed, and memory mixing of different temporal representations. Recently, researchers have used Bayesian inference to analyze contextual calibration and improve performance, but how this links to temporal processing is unclear. The article by [Zhuanghua Shi et al](https://www.sciencedirect.com/science/article/abs/pii/S1364661313002131) reviews the influence of contextual calibration on interval timing, compares Bayesian inference with traditional timing models, and provides a roadmap for integrating the two.

Objective time: For example, adjust the running speed to catch a flying ball. We have no control over time (1 day = 24hrs).  
{{< figure class="center" src="/images/catch.png" >}}

Subjective time: During happy moment, time flies. But during sad time, time goes very slowly. 

{{< figure class="center" src="/images/happy.png" >}}

**Contextual calibration of time perception**

What is the paper by Zhuanghua Shi et al ll about?
Integrate a Bayesian framework with information-processing models of timing to understand the temporal calibration.

* Subjective durations can be easily distorted. 

Example 1: Vierordt’s law (Central-tendency effect): 
Participants overproduce “short” durations and underproduce “long” durations. It is because the duration judgments are derived from both current and the previously experienced stimulus duration. 


Observation: 
1.  String musicians show very low biases (auditory ∆t reproduction).

2.  Expert drummers reproduce ∆t at a near perfection (auditory+visual).

3.  People with Parkinson’s disease. 
    They are prone to contextual manipulation based on medication. Known as temporal “migration” effect.
   
PD with medication OFF tend to overproduce “short” durations and underproduce “long” durations - Central-tendency effect. Following is the graph;
{{< figure class="center" src="/images/pd.png" >}}

Following is the figure that shows, Medication ON/OFF states. Note that the error bars are quite big for OFF medication.
{{< figure class="center" src="/images/error.png" >}}
and simillary the application of Bayesian inference to medication ON/OFF for 21 sec.

In this case, the prior distribution does not change. But with medication OFF state, uncertainty in the likelihood increase. Hence the posterior is shifting to the left. Following is the figure;
{{< figure class="center" src="/images/21.png" >}}

There are other examples in which subjective durations can be easily distorted;

Example 2: Time Order Error (TOE)
This says the order of the presentation matter and it can create a bias in the judgement. 

Example 3:  Modality Effects
Sounds are judged longer than lights. Participants simply overestimate the auditory and underestimate the visual stimuli.

**Traditional approaches to contextual calibration**
1. Adaptation Level (AL) Theory: 
   * According to this theory, a percept of a stimulus depends on the background context. (Luminosity)

   * A modified version of AL theory suggest; perceived subjective duration is a linear weighted average of the sensory evidence and context. 
   
   {{< figure class="center" src="/images/formula1.png" >}}
   This equation explains the Medication ON/OFF states (linear graph) that we have seen above (error bars are quite big for OFF medication).
   
2.  Scalar Timing Theory:  
    This is another theory to describe cognitive process involving in time discrimination.
    Following is an important formula that describe various aspects in the process;
    {{< figure class="center" src="/images/formula2.png" >}}
    
    Following is a schematic for an information-processing (IP) model of scalar timing theory.
    {{< figure class="center" src="/images/IPmodel.png" >}}
    
    According to this theory, standard deviation of the “temporal estimate” increases linearly with the mean of the duration. This property knows as **scalar property**. 
    This linearity comes from the memory translation constant K. Following figure explains it;
    {{< figure class="center" src="/images/subtime.png" >}}
    
    It is important to note that the **Scalar property** also induce memory-mixing (Modality effect). Following figure shows the modality effect;
    {{< figure class="center" src="/images/modality.png" >}}
    
    * Duration bisection procedure using anchor duration of 2 s & 8 s. 
    * Participants were exposed to a range of intermediate durations.
    
    **All participants reproduced the classic finding that “sounds are judged longer than lights”.**

**Bayesian inference on temporal contextual calibration**

* Linear-weighted average model + Scalar timing theory help us to understand memory-mixing  phenomena etc.
* However, it does not explain what factor(s) quantitatively determine the level of contextual calibration.
 
Why Bayesian is good? Because it combines prior knowledge in the statistical distribution. 
What is Bayes' Formula? It is a mathematical formula used in Bayesian statistics that expresses the relationship between the probability of an event (A) given some prior information (B), and the probability of the prior information given the event. It is named after Thomas Bayes, an 18th-century statistician who first described the formula.

The formula is stated as:

	P(D|S) = (P(S|D) * P(D)) / P(S)

where:

  * P(D|S) is the probability of event A given that event B has occurred, also known as the posterior probability.
  * P(S|D) is the probability of event B given that event A has occurred, also known as the likelihood.
  * P(D) is the prior probability of event A occurring.
  * P(S) is the prior probability of event B occurring.

Bayes' Formula is used in Bayesian statistics to update beliefs or probabilities based on new data. It allows for the incorporation of prior information and the computation of posterior probabilities in the light of new evidence. Computing posterior distribution is knowns as the inference problem. 

Similarity between Bayesian inference and information - processing model. 
* P(S|D) ~ N(𝞵_s, 𝞼_s),  Likelihood function of D and 
* P(D) ~ N(𝞵_p, 𝞼_p),  Prior probability of D

{{< figure class="center" src="/images/ip1.png" >}}
By minimizing the loss function L, we obtain posterior mean. Where w_p is the waited variance.
{{< figure class="center" src="/images/mean.png" >}}

**Bayesian inference for predicting central-tendency**

_Jazayeri et al._ published an interesting paper  (Temporal context calibrates interval timing) on it. Where they have choosen partially overlapped intervals.
{{< figure class="center" src="/images/inference.png" >}}

The paper shows how to use Bayesian formula to describe “Central-tendency” effect in temporal reproduction.
{{< figure class="center" src="/images/tendency.png" >}}

The figure illustrate the followings; 
* Production times monotonically increases with sample interval.
* Average production time deviated from the line y = x and towards the mean of the prior (systematic bias). 
* For “long”, deviation is high.. Strong bias.

How Bayesian inference is computed using the Bayes least-square (BLS) estimator?

{{< figure class="center" src="/images/bls.png" >}}

* The mean of the posterior determines the estimate. The resulting mapping function, f_BLS, is sigmoidal in shape. 
* There are other estimators; MLE, MAP.

**Bayesian inference for predicting Modality effect**
* Pulses are integrated at a faster rate for auditory stimuli than for visual stimuli.

* The internal reference of the mean duration between the “short” (S) and “long” (L) anchor durations is larger for auditory stimuli (M_a) than for visual stimuli (M_v).

* M_a and M_v are two independent Gaussians.

* When mixed within the same memory distribution, M_av is a linear-weighted average of M_a and M_v.

{{< figure class="center" src="/images/pse.png" >}}
Here, the PSE is the “point of subjective equality”, when the two stimuli (auditory and visual) look subjectively the same, and thus, an observer would choose randomly between them.
But recall that pulses are integrated at different rate for auditory and visual stimuli.

Integrating Bayesian inference with scalar timing theory.

 - Let's compare key components of both methods.
 
| Similarity | Difference |
|----------|----------|
| Likelihood, the prior probability, and the loss function for optimization can be mapped to the clock, memory, and decision stage.  | Scalar timing theory assume, scalar property comes from memory translation constant. But Bayesian framework does not provide any specific assumption.|
| Clock stage responsible for measurement of external event, while likelihood give insight into physical duration. | Bayesian framework use Baye rule to update the memory, while scalar timing theory uses dynamic memory update. eq-2 |
| Scalar timing theory has 2 memories (working+reference), while Bayesian inference assumes prior and posterior probability distribution. | Preferred loss function in the scalar timing theory is relative error, where as in the Bayesian framework we use squared-error.|

**Concluding remarks**

* This studies shows that the three essential components of a Bayesian framework (i.e., likelihood, prior, and loss function) are closely linked to the clock, memory, and decision stages advocated by scalar timing theory and incorporated into other timing models.

* The Bayesian framework combined with scalar timing theory not only provides a new perspective on interval timing, but also offers quantitative predictions of distortions in temporal memory for normal participants and patients. 

* This integrated system should be combine with the striatal beat–frequency framework to expand our understanding in the mechanistic level.

Visit to see the Python implimentation for central tendancy effect; 
[Bayesian Optimization](https://github.com/AIThoughtLab/Bayesian-optimization)







 

















