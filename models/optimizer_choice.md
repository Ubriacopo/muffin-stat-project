# Hyperparameters
## Optimizer
What optimizer to pick?
> We use this as reference: https://www.lightly.ai/post/which-optimizer-should-i-use-for-my-machine-learning-project#:~:text=Try%20to%20find%20an%20optimizer,previously%20unseen%20data%20%5B14%5D.

The problem with choosing an optimizer is that, due to the no-free-lunch theorem, there is no single optimizer to rule them all; as a matter of fact, the performance of an optimizer is highly dependent on the setting. So, the central question that arises is:

#### Which optimizer suits the characteristics of my project the best?
Adam needs some fine tuning of parameters to really shine.

![alt text](/home/jacopo/Downloads/62cd5ce03261cbb02f18863c_table.png)


## Adam / AdamW
So, while a good choice could be optiming for AdamW as it usually performs well we have to consider the memory overhead it creates in order to compute the training gradients. If the size of the network is large this can lead to real problems as the VRAM of my machine (which is the one used for all this study) is limited to only 8GB.

> Dont use Adam as it may perform randomly. The parameters are too important to be left alone, they need to be fine tuned.
> We had the issue with the loss that was spiking to ~8 and never decreasing. \
> We might have encountered the following problem: https://arxiv.org/abs/2304.09871 (Leggi paper)

### Pro:
- It usually performs well if well tuned

### Con:
- We need to fine tune it

## AdaGrad / AdaDelta
Could be another solid choice. It has the advantage of having fewer tunable parameters and a lower memory impact.
It tends to generalize worse therefore this could be a problem.

Another option would be using Adadelta to improve Adagrad as:
> Adadelta [13] is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size.
> https://www.ruder.io/optimizing-gradient-descent/


## SGD
> Interestingly, many recent papers use vanilla SGD without momentum and a simple learning rate annealing schedule. As has been shown, SGD usually achieves to find a minimum, but it might take significantly longer than with some of the optimizers, is much more reliant on a robust initialization and annealing schedule, and may get stuck in saddle points rather than local minima. Consequently, if you care about fast convergence and train a deep or complex neural network, you should choose one of the adaptive learning rate methods.
> https://www.ruder.io/optimizing-gradient-descent/


From source:
> As a rule of thumb: If you have the resources to find a good learning rate schedule, SGD with momentum is a solid choice. If you are in need of quick results without extensive hypertuning, tend towards adaptive gradient methods.
>https://www.lightly.ai/post/which-optimizer-should-i-use-for-my-machine-learning-project
 
 
 ## SGD with momentum + learning rate schedules
> A common pattern when training deep learning models is to gradually reduce the learning as training progresses. This is generally known as "learning rate decay".
 The learning decay schedule could be static (fixed in advance, as a function of the current epoch or the current batch index), or dynamic (responding to the current behavior of the model, in particular the validation loss).
 S\- keras documentation

Keras has different learning rate schedules already implemented so we could expriment with those.

