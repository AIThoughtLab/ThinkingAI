+++
author = "Mohamed ABDUL GAFOOR"
date = "2020-12-05"
title = "Choregraphe Software for Pepper Robot"
slug = "Choregraphe"
tags = [
    "Social Robotics"
]
categories = [
    "Artificial Intelligence", "Social Robotics"
]

+++

The purpose of this post is to build up a robot speaking system with body language and speech text and speech sound in Choregraphe. See the [documentation here](http://doc.aldebaran.com/2-4/software/choregraphe/index.html) or [download](https://www.robotlab.com/choregraphe-download-page-for-pepper-robot) the software here.

The process is very straight forward using this software. 

Step-1:
Firstly we opened Choregraphe App and connected to the virtual robot 'NAO H21​'. After
that, we selected the 'Say' node, ​'Delay' node and 'Play Sound' ​node on the Box library. In
the ​'Say' node, we put in the words that Pepper will speak. (Here we test with 'NAO H21'
but finally we changed it to ​'Pepper'​). Then go to ​Connection ​and ​Connect to virtual robot
in menu.

{{< figure class="center" src="/images/step1.png" >}}


Add the following nodes in order to create ​text​, ​delay ​and ​sounds ​in the virtual object. The
nodes can be added easy by searching the key words in the search windows (show filters).
The following screenshot shows how this can be accomplished.

{{< figure class="center" src="/images/say.png" >}}

If we want to add the text to the 'say' node, we can simply right click and ​'set parameters' of say. The following screenshot shows how this can be achieved.

{{< figure class="center" src="/images/pep.png" >}}

Step 2:
In order to listen to Pepper speaking, we added several 'Play Sound' nodes in the Box
Library and put the .wav file as parameters. Here we used 'text2speech' to generate sound
files. We only need to type in the words to be converted as voice and download the
generated file. The link of 'text2speech' is provided in reference.

{{< figure class="center" src="/images/text.png" >}}

The following figure shows how an audio file can be added to the Play Sound node.

{{< figure class="center" src="/images/audio.png" >}}


Step 3:
We created as well the 'Delay' node in order to have the text and sound synchronized and
to imitate the natural way of speaking. We tuned the time in each Delay node to achieve it.
Right click on the Delay node and set parameters. Where we can set the time of delay after
each text/audio.

{{< figure class="center" src="/images/delay.png" >}}

Step 4:
At this step, we would create animations that are specified in the provided json file. We
created a python script by creating a new box with right click. The below image shows the
procedure to create a Python script.

{{< figure class="center" src="/images/py.png" >}}

In the __init__ method, add the following lines. This method is called when an object is
created from MyClass. This method helps the class to initialize the attributes of the class.
First we have to create a session using qi.Session(), then add the tcp IP and port. The
default IP is "127.0.0.1".


{{< figure class="center" src="/images/iniy.png" >}}

The port information can be obtained by go to Edit → Preference → Virtual Robot.
{{< figure class="center" src="/images/port.png" >}}


Now in the def onInput_onStart(self) method, add the following line to create the
necessary animation. We must set the path to the json file and load the json file and store it
into the robotpose. If you want to run this project in your local machine, don’t forget to
change the path of json file in the code below. Make sure to set the correct port information.

{{< figure class="center" src="/images/oninput.png" >}}

The following screenshot shows the final implementation of the Choregraphe project.

{{< figure class="center" src="/images/chro.png" >}}

Visit the [Github page](https://github.com/AIThoughtLab/Choregraphe) to see the implementation.


Reference:
1. Text 2 Speech
2. Choregraphe application: NAO6 Downloads - Linux | SoftBank Robotics Developer Center
3. Joint control API — Aldebaran 2.5.11.14a documentation
4. [ALMotionProxy](http://doc.aldebaran.com/2-5/naoqi/motion/control-joint-api.html#ALMotionProxy::angleInterpolation__AL::ALValueCR.AL::ALValueCR.AL::ALValueCR.bCR)







