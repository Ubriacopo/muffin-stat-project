# Statistical Machine Learning Project: Neural Networks
## 1 - Project Request
Use Keras to train a neural network for the binary classification of muffins and Chihuahuas based on images from a dataset provided on Kaggle.

*Images must be transformed from JPG to RGB (or grayscale) pixel values and scaled down*.

The student is asked to:
- Experiment with *different network architectures (at least 3) and training hyperparameters*
- Use 5-fold cross validation to compute your risk estimates
- Thoroughly discuss the obtained results, documenting the influence of the choice of the network architecture and the tuning of the hyperparameters on the final cross-validated risk estimate.

**While the training loss can be chosen freely, the reported cross-validated estimates must be computed according to the zero-one loss.**

## 2 - Considerations on the Request (AR)

#### *Images must be transformed from JPG to RGB (or grayscale) pixel values and scaled down*.

The task itself is simple but as Technical Considerations explain we have to choose the right tradeoff between image resolution and NN complexity or else we run OOM. (Out of mana)

#### Experiment with *different network architectures (at least 3) and training hyperparameters*

To approach the problem in the best way I decided to make these  different structures:

- Naive Approach: Simple DNN (1 Hidden-Layer) (Number of Hidden Layers?)
- Auto Tuned CNN
- Known Architecture: VGG-16
- Known Architecture: LeNet-5

Looke up at : https://towardsdatascience.com/neural-network-architectures-156e5bad51ba

#### Use 5-fold cross validation to compute your risk estimates
We require a way to make 5-CV which will be coded by hand.

## 3 - Technical Considerations 
Given the fact that I spent a lot of money on a Graphic card (NVIDIA RTX 3070Ti) I would love to be able to use it for the task. Thing is that my model has a little amount of VRAM, only 8GB. 
This is for sure a strong limitation on the task as the images can require lots of memory. 

In Keras the model can be quite large during training and the images are loaded in batches so we either have low image resolution or we use a very little batch size.  

## 4 - Other Considerations
### 4.1 - What Optimizer?
In order to speed up convergence, the mini-batched version of stochastic gradient descent is often used.

"We show that for simple overparameterized problems, adaptive methods often find drastically different solutions than gradient descent (GD) or stochastic gradient descent (SGD)"

“We observe that the solutions found by adaptive methods generalize worse (often significantly worse) than SGD, even when these solutions have better training performance. These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks.”
> Reference: https://arxiv.org/abs/1705.08292

Also note that:
> https://paperswithcode.com/paper/on-the-convergence-of-adam-and-beyond-1

Also further proves the point 
> https://proceedings.neurips.cc/paper/2020/hash/f3f27a324736617f20abbf2ffd806f6d-Abstract.html


The choice of using Adam becomes more of a time relevant decision than performance wise.
What we will do now is try both optimizers and see if there is a notable difference in our models.

### 4.2 - How many Epoches?
Number of Epoches are actually an hyperparameter so we could be looking the best possible value. 
> There is no general rule. But you can assign a large number of epochs (let say more than 1000), and then use early stopping regularization. 
> This technique prevents over-fitting by stopping the training procedure, once the model performance on the validation subset does not improve for a certain number of epochs.
> 
> Implementations of early stopping are already provided, check this: https://blog.paperspace.com/tensorflow-callbacks/


### 4.3 - Image Augmentation

What is Augmentation?
> Image augmentation is a process of creating new training examples from the existing ones. To make a new sample, you slightly change the original image. For instance, you could make a new image a little brighter; you could cut a piece from the original image; you could make a new image by mirroring the original one, etc.
> : As defined in https://albumentations.ai/

> Image Augmentation makes only sense during the training phase as we cannot work on the validation dataset as we would loose independence from the procedure to evaluate performance. 


In keras this process is made simple by the layered structure of the Neural Network.
Some of the pre-processing layers defined by keras will be run only during training.

Also remember that:
> *Injecting noise in the input to a neural network can also be seen as a form of data augmentation.*
> — Page 241, Deep Learning, 2016.

#### What size the Images?
Paper lists https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8700246/ as there is no defined metric but a trend for higher resolution images to give better performing CNN. 

Taking 512x512 RGB is unsustainable for my machine as VRAM of the GPU is not enough so we will try with the 256x256 resolution looking for the best achievable result. From there if the process results unfeasable we keep reducing resolution and removing colors on and on.


### 4.4 - The activation function
What do we pick? Boh. This needs better investigation.