# Machine Learning Assortment
In this repository, you will find Machine Learning examples. I try to make the code as clear as possible, and my goal
is that this may be used as a learning resource. If you have any questions, please feel free to be in touch!

## Running models
All projects follow the directory structure:
Project 
- train.py
- pipeline.py
- param.py (optional)
- util.py (support files)

I chose this structure so a project may be run on the command line
```commandline
export PYTHONPATH="path/to/Machine-Learning-Assortment"
python (project)/train.py
```

## Convolutional Neural Network
A Convolutional Neural Network (CNN) mimics the process of the human eye. Vision in the human eye is dependent on rods and cones, where rods are 
responsible for nightvision and cones for the day time. There are three types of cones, which allows the eye to see red, green, blue, and mixes 
of those colors. There are roughly 4.5 million cones in the eye, most located in the fovea with 150,000 cones / mm**2. There are about 100
million rods in each eye. The rods, responsible for night vision, connect to neurons in a many-to-one relationship. The cones, responsible for 
color in bright light, connect to neurons differently. One cone connects to many neurons in a one-to-many relationship. For night vision, 
there is not much light and the multiple receptors are needed to weigh in before the brain sees light. For day vision, there's a lot of light and 
one rod may see something vastly different than its neighbor, so multiple neurons advocate for its cone and communicate in a network 
as to what the eye just saw. Stanford provided good background on the eye in [Foundations of Vision](https://foundationsofvision.stanford.edu/chapter-3-the-photoreceptor-mosaic/). 


Let's now look at a convolution in terms of an image. Instead of rods, pixels will be our input. Instead of a vista seen by the eye, let this 
be a photo from a camera. A convolution is a way to reduce the image to get out the important information. 
A convolution is a sliding window, usually square, which selects pixels from the image. The selected pixels then
go through an operation or function which results in a single value. The window slides, performs the function again, and returns a new value. 
This continues, creating a smaller image. More than size reduction, (which is important, one does enjoy to have low-res photos match with
high-res) convolution functions can increase contrast, change colors, and even be used to classify faces in a photo. The function is key. The function
of a convolution can be anything, which is why it is so generically named function. Meant to be general, the function can take the max, mean, min,
cross product, log, and any mix of them. For CNNs, the neural network part is the function. Each function is per each window the pixels
just do matrix multiplication on a set of values, let's call them weights. The weights change, which means the function changes. As the 
sliding window multiplies the weights, the generated image changes, which creates a chain reaction of changes. In the CNN, the final result
is compared to the target result. If they match, that's great. If they miss, then how far apart they are is measured and the weights are adjusted. 

## U-Net
A U-Net, from the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), is a useful way to segment an image. Often people label parts of an
image by hand, such as regions in the human brain. Labelling by hand takes a long time, and it's worthwhile to use a computer to 
do this for you. The U-Net was a breakthrough in 2015 for computer vision, winning the ISBI tracking challenge. Variants of U-Net continue to be 
state of the art. 

The U-Net performs semantic segmentation. This means the U-Net can classify categories of objects, like bicycles and street signs, but will not 
identify each bike or each street sign. To segment objects each with an identity is instance segmenation which may be done with
Mask-RCNN or YOLO architectures. The U-Net is built with convolutional layers. As you progress through the network, the layers shrink in height
and width, but increase in depth. By depth, I mean the third dimension of the layer; if the layer were an image then depth would be its channels,
red, green, and blue, but instead of three colors, there would be 64 or 128. This means as the image is processed, it transforms into 
stacks and stacks of images, all containing information. Increasing the depth of each layer increases the complexity of data the layer
can hold and, if you need it detect complex patterns, then that is perfect for your problem. The U-Net is structure in groups of three, three convolutions, then a 
2x2 max-pooling layer. Each three layers of convolutions forms a convolutional block. 
The convolutions reduce the height and width by two pixels for each layer. The max-pooling
layer reduces the layer's height and width by two, and in consequence its area by four. 

The smallest layer in the U-Net has a height and width
of 30, with a depth of 1024. This would be like transforming an image of 572x572 with one channel to an image of 30x30x1024. Interesting to 
imagine, isn't it? The goal is get a segmented image out of this, binarized and the same size as the 
input image. So, as we progress through the net, the layers actually increase in height and width, but the depth diminishes. In the paper, in
order to upsample, the authors used interpolation to fill in the blanks. As the net upsamples, weights continue to transform the layers and it 
something very different than even a few convolutional blocks ago. It became a problem that the U-Net would learn something learn, but
then the lesson would be forgotten, overwritten, and lost. In order to overcome this setback, the shrinking convolutional blocks are paired with
the growing convolution blocks and the half of the shrinking layer is copied directly to the growing layer. This trick is useful in deep learning. 
Copying part of layers is also used in ResNet architecture. It makes the deep learning model easier to train because it avoids 
catastrophic forgetting.

## Multiple Instance Learning
Multiple Instance Learning (MIL) is a machine learning technique used when labels are soft. These datasets do not have exact labels, but rather 
groups of objects are labelled, and some may be wrong. This is particularly useful in digital pathology, where whole biopsy slides are labelled, 
but the phenotypic cells driving the pathologists' diagnosis is not. Let each object be called an instance and each group of instances be 
called a bag. For binary classification, MIL randomly assembles bags of instances and if there is at least on positive instance in the bag, then that bag is positive. If 
all the instances in the bag is negative, then the bag is negative. As a thought experiment, imagine there are multiple people
with sets of keys on a keychain. Some of the keychains open a door and some keychains don't. You are tasked with finding the 
key that opens the door. Using MIL, keys are switched on the keychains and tested until one key is determined to be the key to
open the lock. 

## Deep Convolutional Generative Adversarial Networks (DCGAN)

### Generative Adversarial Network (GAN)
A Generative Adversarial Network, first introduced by [Ian Goodfellow in 2014 as "Generative Adversarial Nets"](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
is a deep learning architecture that is trained on a dataset to generate similar data. One use could be to expand your training
dataset. Advanced variations of GANs the generation of AI art. In a Generative Adversarial Network, there are two models. One is a 
Generator. This generates new data. The adversary is called the Discriminator, which discerns whether the data it is examining
is real or generated. In the end, we expect the Discriminator model to predict the samples with 0.5 accuracy, showing it is guessing 
with random chance. 

### DCGAN
The Deep Convolutional Generative Adversarial Network is a variant of GAN, built with convolutional layers in the discriminator and 
convolutional-transpose layers in the generator. The paper, [Unsupervised Representation 
Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf), lists the major changes
the authors determined necessary in order to make the DCGAN successful. The discriminator is designed with strided convolutional layers, which means the stride of the 
convolution was at least two, which replaced deterministic pooling functions like max pool. The use of convolutional layers allows the model to learn
its downsampling rather than force the downsample to be the maximum value in a window. It was debated whether to use fully connected layers 
after the convolutions or global average pooling, which has been shown to improve performance in some models. But, the authors didn't use either 
and flattened the convolutional-transpose layer of the generator and fed it directly into the discriminator. The model does not use pooling or 
fully connected layers. Batch normalization, which scaled the input to have zero mean and unit variance, was heavily relied on. Without it, the 
DCGAN did not learn. However, when batch norm was on every layer, learning was unstable so the final layer of the generator and the
input layer of the discriminator do not have batch norm. The paper tested different activation functions and chose Tanh for the output layer, 
because it was bounded and learned more quickly. In the generator, the activation ReLU was chosen and in the discriminator
Leaky ReLu worked well. 

#### Loss Function
In each batch, the discriminator must be run twice, once for real data and once for generated data output from the generator. 
The outputs are then compared with binary crossentropy loss. Entropy, in general, a measure of disorder or randomness in a system. In
information theory, disorder or randomness in a message or signal is uncertainty. A message or signal with low entropy is accurate. In 
thermodynamics, entropy in a system measures the energy available to do work, where work refers to energy transferred from one system to another
causing the state of the system to change such as physical shape or the generation of electricity. If the entropy of a thermodynamical system is
high, the energy is spread out and heat is evenly distributed. If the entropy of a system is low, then energy is very concentrated among a few 
particles in the system. In thermodynamics, energy is not free floating, it is contained within particles. Even energy of light is manifest through photons
with zero mass. If the energy of the system is low, the only a few particles have high amount of energy and the rest of the particles have low energy,
which means some particles are moving quite fast and others are not. If the particles crash into each other sufficiently, then energy will be transferred,
the state of the system will evolve, and entropy will increase. Imagine the particles like bins in a histogram where the bins are filled with energy, creating
a probability distribution. Entropy is based on probability which is how it applies to computer science, statistics, information theory, and physics. 

## Deep Q-Networks (DQN)
This repo shows an example of training Cartpole from the Gymnasium library to balance. Deep Q-Learning is an extension of Q-Learning that uses a deep neural network rather than a table to store the action-value function. Q-Learning  
is a model-free reinforcement learning algorithm. Model-free means that the algorithm does not need a model of the environment, such as 
to know all possible moves a player can make. Q-Learning has vocabulary, which is critical to know in order to understand it. For example, this sentence
makes sense: Given a state, the agent takes actions based on a policy to maximize rewards in the dynamic environment. DQNs were 
popularized by the paper, [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), showing their success on Atari games, such as Space Invaders. 

The agent is the brain, the guide, it's like the computer replacing the human player. In this case the Cartpole from Gymnasium is the agent. In Space Invaders, the sIn mario, the agent is mario.
Actions are what the player can do. The cartpole, in order to balance, can move left and right. A reward is what it sounds like: the player is doing well and accomplishing their goal and that is noted by the
training algorithm. In the Cartpole example, the pole's reward is based on how long it can balance. The environment is what the player interacts with. For the Cartpole, the 
environment is just an infinite space extending horizontally. The pole can move left and right with not obstructions. The state is the current characteristics of the player and environment. For the Cartpole, 
the state is the angle of the pole, the gravitational pull, and its momentum. The policy is what guides the actions of the agent. Based on the state,
the policy is the function that tells the agent what action to take. If the Cartpole is leaning too far to the right, the policy should compensate
by telling to pole to move quickly right in order to balance.
