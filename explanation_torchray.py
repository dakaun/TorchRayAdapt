import torchvision
import torch
import argparse
import os
from PIL import Image

from torchray.benchmark import datasets, models, plot_example
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient
from torchray.attribution.meaningful_perturbation import train_mask
from torchray.attribution.extremal_perturbation import extremal_perturbation, contrastive_reward
from torchray.attribution.deconvnet import deconvnet

if __name__ == '__main__':
    '''
    Method to try visualization methods on your data. Either on a specified example image ('flute.jpg'), on a common
    dataset or specify an own dataset. 
    If a dataset is chosen, then transfer learning is utilized to adapt the specified model (with imagenet weights) to
    the defined dataset. As transfer learning, a feature extractor is used which takes the trained conv model and only
    retrains the last layer while treating the rest of the conv model as a fixed feature extractor. 
    '''
    parser = argparse.ArgumentParser(description='TorchRayAdapt. Choose between different visualization methods for '
                                                 'your data.')
    parser.add_argument('--img', type=str, default=None,
                        help='specify path to example image. Alternatively specify a dataset')  # 'flute.jpg'
    parser.add_argument('-d', type=str, default='cifar', choices=['cifar', 'imagenet', 'stl10', 'own'],
                        help='specify a dataset')  # choose own if an own dataset is available in ImageFolder
    parser.add_argument('-e', type=str, default='grad_cam', choices=['deconvnet', 'grad_cam', 'gradient',
                                                                     'extremal_perturbation',
                                                                     'meaningful_perturbation'],
                        help='specify the explanation method')
    parser.add_argument('-m', type=str, default='vgg16', choices=['vgg16', 'resnet50'],
                        help='specify the model architecture')
    args = parser.parse_args()
    dataset_name = args.d
    expl_method = args.e
    modelarch_name = args.m

    if args.img is not None:
        own_image = True
        image_path = args.img
        print(f'Chosen explanation method: {expl_method}, \nModelname: {modelarch_name}, \nImage path: {image_path}')
    else:
        own_image = False
        print(f'Chosen explanation method: {expl_method}, \nModelname: {modelarch_name}, \nDataset: {dataset_name}')

    shape = 224  # defined by model
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(shape),
        torchvision.transforms.CenterCrop(shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    if own_image:
        labels = [558]
        original_img = Image.open(image_path)
        images = transform(original_img).unsqueeze(0)

        model = models.get_model(arch=modelarch_name,
                                 dataset=dataset_name  # model for specified dataset
                                 )
        classnames = datasets.IMAGENET_CLASSES
    else:
        # DATASET
        dataset, classnames = datasets.get_dataset(name=dataset_name,
                                                   subset='train',  # train oder val
                                                   # dataset_dir= ,  # define if own dataset in ImageFolder format
                                                   download=True,
                                                   transform=transform
                                                   )
        dataset_split = torch.utils.data.Subset(dataset, indices=range(40))
        dataloader = torch.utils.data.DataLoader(dataset_split, batch_size=4, shuffle=True, num_workers=4)

        # MODEL
        model = models.get_model(arch=modelarch_name,
                                 dataset=dataset_name,  # model for specified dataset
                                 nb_classes=len(classnames)
                                 )

        # TRANSFER LEARNING (feature extractor)
        model, criterion, optimizer_conv, exp_lr_scheduler = models.transfer_learning_prep(model,
                                                                                           modelarch_name,
                                                                                           len(classnames))

        model_path = './models/'
        if not os.path.exists(model_path): os.mkdir(model_path)
        model_path = model_path + dataset_name + '_on_' + modelarch_name + '.pth'

        if not os.path.exists(model_path):
            model = models.train_model(model, criterion, optimizer_conv, exp_lr_scheduler,  # train_model
                                       dataloader, len(dataset_split), epochs=15)
            torch.save(model.state_dict(), model_path)  # save model
        else:
            model.load_state_dict(torch.load(model_path))  # load trained model
            # models.visualize_model(model, dataloader, classnames, num_images=6)

        # EXPLANATION METHOD
        images, labels = iter(dataloader).next()

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
    elif expl_method == 'extremal_perturbation':
        saliency, _ = extremal_perturbation(
            model=model,
            input=images,
            target=labels[0],
            reward_func=contrastive_reward,
            debug=True,
            areas=[0.1]
        )
    elif expl_method == 'deconvnet':
        saliency = deconvnet(
            model=model,
            input=images,
            target=labels
        )
    else:
        assert False, 'Explanation Method is not yet defined'

    plot_example(images, saliency, expl_method, labels, classnames, show_plot=True,
                 save_path=expl_method + '_' + classnames[labels[0]] + '.png')
