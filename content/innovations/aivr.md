+++
author = "Mohamed ABDUL GAFOOR"
date = "2020-10-13"
title = "Enhancing human abilities in the age of AI and VR."
slug = "dream"
tags = ["Machine Learning", 
	"Deep Learning", 
	"Virtual Reality"]
categories = [
	"Bayesian Optimization"]

+++


**Visualizing a dream in the VR environment**

Can we ever have the ability to visualize our dreams again after waking up? I often dream of re-watching my dreams after I wake up from sleep. Is this technologically feasible? What are the obstacles in building a system where a person can replay the dream they had last night? I plan to present a simple idea to illustrate this concept.

Dream visualization after waking up is an interesting and complex topic. Currently, there is no technology that allows people to fully visualize their dreams after they wake up. This is due to several reasons, including the elusive nature of dreaming and the lack of scientific understanding of the underlying processes.

Dreams are thought to occur in the brain during rapid eye movement (REM) sleep, which is associated with deep sleep and vivid dreaming. During this stage of sleep, the brain is highly active and generates complex patterns of neural activity. However, these patterns of neural activity are not easily accessible and cannot be directly recorded or monitored.

Additionally, the subjective nature of dreaming makes it difficult to translate the experience into a form that can be objectively visualized. Dreams are personal and unique to each individual, and there is no universally accepted way to translate a dream into a visual representation.

Despite these challenges, some researchers and scientists are exploring different approaches to dream visualization. For example, some are using brain-computer interfaces to try to decode the neural signals generated during REM sleep and recreate the dream experience. 

A quick review on how we see things in our environment

Light enters into eye through the cornea and finally reaches the retina. Retina is a light-sensitive nerve layer where the image is inverted. The optic nerve is then carry the signals to the visual cortex of the brain. Whenever we see objects/images a set of neurons generates spikes in the visual cortex. This phenomena can be recorded via Electroencephalography (EEG) or neural interface microsystem: 100-element silicon-based MEA via a custom hermetic feedthrough design (Ref: 1). 

**Illustration**

To further illustrate, let's start with a simple image of a cat. When we see a cat, a set of neurons in the brain are activated. It doesn't matter how many times we see the cat, more or less the same sets of neurons/areas will be activated in the brain. This signal must be transformed into a computer, and machine learning/deep learning techniques must be used to train a system that contains a lot of neural activity data for a specific image. So whenever an image is presented in front of the eye, after training the neural network (NN), it will display a possible image on the computer screen (cat/dog). Let us create a Python script to generate a random EEG data for cat/dog. 

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random EEG data
fs = 128 # Sampling frequency (Hz)
t = np.arange(0, 10, 1/fs) # Time vector (10 seconds)
x = np.random.randn(len(t)) # Random EEG data

# Plot the EEG data
plt.plot(t, x, color='red')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.title('Example EEG Data for Cat')
plt.show()
```

The following table is a simplified version that can be used to train the NN for known objects and neural activity data.
{{< figure class="center" src="/images/catdog.png" title="Figure: Brain signals and corresponding objects.">}}

This process is very complicated than I present here. But our purpose is to model a simplified version in a theoretical sense. Now if we could predict the object based on the brain signal, this object can be placed in a VR environment. Not necessarily the exact cat/dog we saw in the dream, but the 3D reconstruction of a similar cat/dog. Of course we can reproduce the exact cat/dog we saw in the dream. But our first priority is to recognize the object in the first place while someone is dreaming in her/his sleep. 

According to Scientific American, it is indeed possible to measure brain activities during the time of dreaming. I will go one step further and try to understand the content of the dream and re-visualize in the VR environment. Hence, I will use the Idea 1 to build a prototype, a sketch that can be used to build the whole system in the future. This is a simplified testing, in which we use only two images cat/dog. 

* Step 1: Take several brain neural activity data by showing images of cats/dogs to participants. 

* Step 2: Carefully record the spikes using EEG or any other nano electrodes.

* Step 3: Map between the images of cats/dogs with the corresponding brain waves. Pixel level mapping will help to reconstruct the exact image.

* Step 4: Train the ML model for the possible prediction of a new (unknown) brain signal.

* Step 5: After obtain the predicted image (cats/dogs) from the brain data, a 3D image reconstruct can be used to obtain a 3D model and game engine like Unity can be used to develop a VR model. 

* Step 6: Immersive visualization of the dream in the VR environment. 

Below figure describe an architectural design of visualizing an object in the VR environment directly from the neural data. 
{{< figure class="center" src="/images/me.png">}}

It is a schematic sketch to reproduce the image from neural activity data. As we explained before, we take neural activity data when the person is awake by showing different images. When the person is sleep, we measure brain activity and try to map the dream based on the obtained brain data.

This is a primitive architectural design to implement and visualize the dream. I assume my world consists of only cats and dogs, no other objects. This assumption is to simplify the model solving ability in a constrained environment before extending to a large scale. There are many obstacles to implement this idea. First of all neuroscience is a still developing field and many mysterious/unanswered questions out there about the function of neurons and brain activity, memory in general. Well this doesn't stop us achieving the goal!! For example, we do not know how exactly water (H2O) molecules behave in a complex fluid flow, we still didn't solve analytically the equation of Navier–Stokes (which is one of the millennium-problems). However, we are good at solving fluid problems numerically using computers, and to forecast the behavior of fluids for the short time interval. Similarly it is not necessary to understand the brain functions fully before we interact with the brain in the neural level. Right technology with right people will help me to achieve this goal one day.

References: 

* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3638022/ 
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3306444/ 
* https://link.springer.com/chapter/10.1007/978-3-642-45062-4_63 
* https://www.scientificamerican.com/article/brain-activity-during-sleep-can-predict-when-someone-is-dreaming/ 
* https://www.leeloosesotericorner.com/dreamsymbol-analysis.html 
* https://www.arxiv-vanity.com/papers/1804.06375/ 
