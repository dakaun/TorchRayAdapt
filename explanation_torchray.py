import torchvision
import torch
from torchray.benchmark import datasets, models
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import os

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
classnames = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer_conv = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1) ## Decay LR by a factor of 0.1 every 7 epochs

# train_model

model = models.train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloader, len(dataset_split), epochs=5)

models.visualize_model(model, dataloader, classnames, num_images=6)
# save model
model_path = './models/'
if not os.path.exists(model_path): os.mkdir(model_path)
torch.save(model.state_dict(), model_path + 'cifar_on_resnet.pth')
# load trained model
model.load_state_dict(torch.load(model_path + 'cifar_on_resnet.pth'))




'''
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# visualize model
def visualize_training(model, num_img=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            _, preds = torch.max(outputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_img//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {classnames[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_img:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
'''