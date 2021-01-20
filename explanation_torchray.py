import torchvision
import torch
from torchray.benchmark import datasets, models, plot_example
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import os

from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient

data = {
    'Cifar': 'cifar',
    'Imagenet': 'imagenet', # no longer publicly available
    'VocDetection': 'voc_2007', #todo update download function
    'Coco': 'coco' # todo test
}

model_archs = {
    'VGG': 'vgg16',
    'Resnet': 'resnet50'
}

expl_methods = {
    #'center',
    #'contrastive_excitation_backprop',
    #'deconvnet',
    #'excitation_backprop',
    'Gradcam': 'grad_cam',
    'Gradient': 'gradient'
    #'guided_backprop',
    #'rise',
    #'extremal_perturbation'
}
dataset_name = data['Cifar']
modelarch_name = model_archs['Resnet']
expl_method = expl_methods['Gradient']

classnames = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #todo integrate class names somewhere

# DATASET
# Pre-process the image and convert into a tensor
shape= 224 # defined by model
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(shape),
    torchvision.transforms.CenterCrop(shape),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

dataset = datasets.get_dataset(name=dataset_name,
                               subset='train', # train oder val
                               download=True,
                               transform= transform
                               )
dataset_split = torch.utils.data.Subset(dataset,indices=range(5000))
dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=64, shuffle=True, num_workers=4)

# IMAGE + CATEGORY_ID
image, label = dataset.__getitem__(np.random.randint(0,len(dataset.data)))


# MODEL
model = models.get_model(arch=modelarch_name,
                         dataset = dataset_name # model for specified dataset
                  )

# PREDICTION
# model(image) # todo mismatch of classes. imagenet 1000, cifar 10 (?)
# transfer learning https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor

# transfer learning
# try convnet as feature extractor, cause vgg much smaller than image net
model, criterion, optimizer_conv, exp_lr_scheduler = models.transfer_learning_prep(model)

# train_model
# model = models.train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloader, len(dataset_split), epochs=5)

# save model
model_path = './models/'
# if not os.path.exists(model_path): os.mkdir(model_path)
# torch.save(model.state_dict(), model_path + 'cifar_on_resnet.pth')

# load trained model
model.load_state_dict(torch.load(model_path + 'cifar_on_resnet.pth'))

models.visualize_model(model, dataloader, classnames, num_images=6)

# todo Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 3-dimensional input of size [3, 224, 224] instead
if expl_method == 'grad_cam':
    grad_cam_layer = '' #todo add layername # features.29'
    saliency = grad_cam(
        model = model,
        input = image,
        target = label,
        saliency_layer = grad_cam_layer
        # resize= ?
    )
elif expl_method == 'gradient':
    saliency = gradient(
        model = model,
        input = image,
        target = label
        # resize = ?
        # smooth = 0.02
    )
else:
    assert False

plot_example(image, saliency, expl_method, label)

