# TorchRayAdapt

This repository is an extension of the TorchRay Repository from Ruth C. Fong and Andrea Vedaldi 
(https://github.com/facebookresearch/TorchRay).
I've created one script which enables to easily choose from different models and datasets to display all available 
explanation methods.
Available explanation methods are those available in the original repository and additionally meaningful perturbations
(https://github.com/jacobgil/pytorch-explain-black-box , Paper: https://arxiv.org/abs/1704.03296).
All methods concentrate on visualization methods for CNNs using PyTorch and range from backpropagation methods
to perturbation methods. 
A full documention with an overview on all methods can be found 
[here](https://facebookresearch.github.io/TorchRay).

**Dependencies**  
This repository uses python 3.8.5 and all further dependencies are listed in requirements.txt

**How to use:**  
The script I wrote is *explanation_torchray.py*. 
In the file all available models and datasets are listed and you can select the most convienient 
combination for your project.
Besides to common data sets (as e.g. CIFAR), you can also specify your own dataset using the Torch ImageFolder or 
select only one image from your available data. 
In addition to the selected model and data, different explanation methods can be tested.

**Examples**  
From TorchRay there are further usage examples in the 
[`examples`](https://github.com/facebookresearch/TorchRay/tree/master/examples)
subdirectory.