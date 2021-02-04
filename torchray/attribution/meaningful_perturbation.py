import numpy as np
import cv2

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def preprocess_image(img):
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img)
    img_tensor.unsqueeze_(0)
    return Variable(img_tensor, requires_grad=False)


def train_mask(model, input, targets):
    '''
    # paper:
    Algorithm from 'Interpretable Explanations of Black Boxes by Meaningful Perturbation'
    (https://arxiv.org/abs/1704.03296). It makes a mask of pixel that explain the result of a black box. The mask is
    learned by posing an optimization problem and solving directly for the mask values.
    Algorithm is from https://github.com/jacobgil/pytorch-explain-black-box
    Args:
        model: model
        input: input torch (image)
        targets: target label

    Returns:

    '''
    inputi = input[0].numpy()
    inputi = np.transpose(inputi, (1, 2, 0))  # change to channel last for blurring
    target = targets[0]

    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 500
    l1_coeff = 0.01
    tv_coeff = 0.2

    blurred_img1 = cv2.GaussianBlur(inputi, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(np.asarray(inputi * 255, dtype=np.uint8),
                                             11)) / 255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2  # todo brauch ich das überhaupt
    mask_init = np.ones((28, 28), dtype=np.float32)

    img = preprocess_image(inputi)
    blurred_img = preprocess_image(blurred_img2)
    mask = numpy_to_torch(mask_init)

    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=(56, 56)).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
    optimizer = torch.optim.Adam([mask], lr=learning_rate)

    for i in range(max_iterations):
        if i%100==0:
            print('Iter ', i)
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                  upsampled_mask.size(3))
        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)

        noise = np.zeros((224, 224, 3), dtype=np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = numpy_to_torch(noise)
        perturbated_input = perturbated_input + noise

        outputs = torch.nn.Softmax()(model(perturbated_input))  # todo is transfer learning  model a problem ?

        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
               tv_coeff * tv_norm(mask, tv_beta) + outputs[0, target]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask.data.clamp_(0, 1)

    upsampled_mask = upsample(mask)

    # prepare mask for resulting image
    mask = upsampled_mask.cpu().data.numpy()[0]
    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask)
    return mask
