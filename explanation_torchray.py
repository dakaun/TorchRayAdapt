import torchvision
import torch
from torchray.benchmark import datasets, models, plot_example
import os

from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient
from torchray.attribution.meaningful_perturbation import train_mask


data = {
    'Cifar': 'cifar',
    'Imagenet': 'imagenet',
    'STL10': 'stl10'
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
    'Gradient': 'gradient',
    #'guided_backprop',
    #'rise',
    #'extremal_perturbation'
    'Meaningful_perturbation' : 'meaningful_perturbation'
}
dataset_name = data['Cifar']
modelarch_name = model_archs['VGG']
expl_method = expl_methods['Meaningful_perturbation']

# DATASET
shape = 224  # defined by model
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(shape),
    torchvision.transforms.CenterCrop(shape),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

dataset, classnames = datasets.get_dataset(name=dataset_name,
                               subset='train',  # train oder val
                               download=True,
                               transform= transform  # transformation dependent on dataset, right?
                               )
dataset_split = torch.utils.data.Subset(dataset,indices=range(40))
dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=4, shuffle=True, num_workers=4)

# MODEL
model = models.get_model(arch=modelarch_name,
                         dataset=dataset_name  # model for specified dataset
                  )

# TRANSFER LEARNING (feature extractor)
model, criterion, optimizer_conv, exp_lr_scheduler = models.transfer_learning_prep(model,
                                                                                   modelarch_name, len(classnames))

model_path = './models/'
if not os.path.exists(model_path): os.mkdir(model_path)
model_path = model_path + dataset_name + '_on_' + modelarch_name + '.pth'

if not os.path.exists(model_path):
    model = models.train_model(model, criterion, optimizer_conv, exp_lr_scheduler,  # train_model
                               dataloader, len(dataset_split), epochs=15)
    torch.save(model.state_dict(), model_path)  # save model
else:
    model.load_state_dict(torch.load(model_path))  # load trained model
    models.visualize_model(model, dataloader, classnames, num_images=6)

# EXPLANATION METHOD
images, labels = iter(dataloader).next() # todo aus imagefolder laden
# todo einzelnes image einfügen.
if expl_method == 'grad_cam':
    grad_cam_layer = ''
    saliency = grad_cam(
        model=model,
        input=images,
        target=labels,
        saliency_layer=grad_cam_layer
    )
elif expl_method == 'gradient':
    saliency = gradient(
        model=model,
        input=images,
        target=labels
    )
elif expl_method == 'meaningful_perturbation':
    saliency = train_mask(
        model=model,
        input=images,
        targets=labels
    )
else:
    assert False, 'Explanation Method is not yet defined'

plot_example(images, saliency, expl_method, labels, classnames, show_plot=True,
             save_path=expl_method + '_' + classnames[labels[0]]+ '.png')

