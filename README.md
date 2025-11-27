# COMP6704 Group Project - Unconstrained optimization problems in deep learning

This is the code repo for the Group 9 project of COMP6704 Advanced Topics in Optimization. The topic of our project is **unconstrained optimization problems in deep learning**.

Our task focuses on handling image classification problems (SVHN, CIFAR10, CIFAR100) by applying standard deep neural networks like ResNet-18 and VGG-16. The optimization procedure is searching optimum model parameter vector via a given loss function by dominant first-order optimizers in the field of deep learning including SGD, NoisySGD, Adam, and AdamW.

Project structure:
```
.
├── Adam_based.py: Training with Adam and AdamW, including the visualization.     
├── model.py: The model structure of ResNet-18 and VGG-16.     
├── nsgd_trainer.py: Training with noisy SGD.    
├── picker.py: The file for datasets and models choices.        
├── ploter.py: The file for plotting the training curves.  
├── sgd_trainer.py: Training with SGD.    
└── README.md  
```


Use the following command to train the model with SGD and noisy SGD:    
```
python sgd_trainer.py
python nsgd_trainer.py
```    

Use the following command to plot the comparison curves of two SGD-based optimizers:     
```
python ploter.py
```   

Use the following command to train the model with Adam and AdamW:

```
python Adam_based.py
```    
or with parameters:     
```
python Adam_based.py --datasets cifar10 --models resnet
```   