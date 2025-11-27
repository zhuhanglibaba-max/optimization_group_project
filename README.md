# optimization_group_project
Unconstrained optimization problems in deep learning
Our task focuses on handling image classification problems (SVHN, CIFAR10, CIFAR100) by applying standard deep neural networks like ResNet-18 and VGG-16. The optimization procedure is searching optimum model parameter vector via a given loss function by dominant first-order optimizers in the field of deep learning including SGD, NoisySGD, Adam, and AdamW.



.
├── Adam_based.py: The file of training with Adam and AdamW. You can run it by commande line directly, with the parameters provided at the end of this file. 
├── model.py: The model structure of ResNet-18 and VGG-16.
├── nsgd_trainer.py: The execution file of training with noisy SGD.
├── picker.py: The file for optimizer choice.
├── ploter.py: The file for plotting the training curves.
├── sgd_trainer.py: The file of training with SGD.
