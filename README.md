# optimization_group_project
Unconstrained optimization problems in deep learning
Our task focuses on handling image classification problems (SVHN, CIFAR10, CIFAR100) by applying standard deep neural networks like ResNet-18 and VGG-16. The optimization procedure is searching optimum model parameter vector via a given loss function by dominant first-order optimizers in the field of deep learning including SGD, NoisySGD, Adam, and AdamW.


├── Adam_based.py: Training with Adam and AdamW. You can run it directly, or using command line with the parameters provided in the file.  
├── model.py: The model structure of ResNet-18 and VGG-16.  
├── nsgd_trainer.py: Training with noisy SGD. You can run it directly.  
├── picker.py: The file for datasets and models choices.  
├── ploter.py: The file for plotting the training curves.  
├── sgd_trainer.py: Training with SGD. You can run it directly.    

You can using the following command line to run the project:
`python Adam_based.py`  
or with parameters, here is an example:  
`python Adam_based.py --datasets cifar10 --models resnet`  
`python nsgd_trainer.py`  
`python sgd_trainer.py`  
