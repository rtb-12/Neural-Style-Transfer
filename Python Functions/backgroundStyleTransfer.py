# Example usage:
input_image_path = r'D:\Neural-Style-Transfer\Images\dancing.jpg'
output_background_image_path = r'D:\Neural-Style-Transfer\Images\background.png'
output_foreground_image_path = r'D:\Neural-Style-Transfer\Images\subject.png'

from rembg import remove
from PIL import Image
import numpy as np


def extract_background_foreground(input_path, output_background_path, output_foreground_path):
    # Load input image
    input_image = Image.open(input_path)
    print("Input image loaded.")

    # Remove background
    output_image = remove(input_image)
    print("Background removed.")

    # Convert images to NumPy arrays
    input_rgb = np.array(input_image)[:, :, :3]  # Extract RGB channels.
    output_rgba = np.array(output_image)  # Output image in RGBA.

    # Extract alpha channel
    alpha = output_rgba[:, :, 3]
    alpha3 = np.dstack((alpha, alpha, alpha))  # Convert to 3 channels

    # Calculate background without subject
    background_rgb = input_rgb.astype(np.float64) * (1 - alpha3.astype(np.float64) / 255)
    background_rgb = background_rgb.astype(np.uint8)  # Convert back to uint8

    # Convert background to PIL image
    background = Image.fromarray(background_rgb)
    print("Computed background without subject.")

    # Extract and save subject (main object)
    subject_rgb = input_rgb * (alpha3 / 255)
    subject_rgb = subject_rgb.astype(np.uint8)  # Convert to uint8

    # Convert subject to PIL image
    subject = Image.fromarray(subject_rgb)
    print("Extracted subject.")

    # Save output images
    background.save(output_background_path)
    subject.save(output_foreground_path)
    print("Output images saved.")



extract_background_foreground(input_image_path, output_background_image_path, output_foreground_image_path)




#Model for style transfer function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image


import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=200,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def style_transfer(content_image_path, style_image_path, output_image_path):
    content_img = Image.open(content_image_path)
    style_img = Image.open(style_image_path)

    # Resize content and style images to have the same dimensions
    content_img = content_img.resize((256, 256))
    style_img = style_img.resize((256,256))

    content_tensor = transforms.ToTensor()(content_img).unsqueeze(0).to(device)
    style_tensor = transforms.ToTensor()(style_img).unsqueeze(0).to(device)
    input_tensor = content_tensor.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_tensor, style_tensor, input_tensor)

    output_image = transforms.ToPILImage()(output.cpu().squeeze(0))
    output_image.save(output_image_path)





content_image_path = output_background_image_path
style_image_path = r"D:\Neural-Style-Transfer\Images\picasso.jpg"
output_image_path = r"D:\Neural-Style-Transfer\Images\BackgroundStyle.jpeg"

style_transfer(content_image_path, style_image_path, output_image_path)

#merge background  and foreground

from PIL import Image
import numpy as np

def remove_background_advanced(image):
    # Convert the image to RGBA mode
    image = image.convert("RGBA")

    # Convert PIL image to numpy array
    data = np.array(image)

    # Set threshold for black pixels
    lower_black = np.array([0, 0, 0, 255])
    upper_black = np.array([10, 10, 10, 255])

    # Create a mask to identify black pixels
    mask = np.all(data[:, :, :3] <= upper_black[:3], axis=-1) & np.all(data[:, :, :3] >= lower_black[:3], axis=-1)

    # Apply the mask to remove black background
    data[mask] = [0, 0, 0, 0]  # Set black pixels to transparent

    # Convert the numpy array back to a PIL image
    image = Image.fromarray(data, 'RGBA')
    return image

def merge_images(background_path, foreground_path, output_path):
    # Open the background and foreground images
    background = Image.open(background_path)
    foreground = Image.open(foreground_path)

    # Remove black background from the foreground image
    foreground = remove_background_advanced(foreground)

    # Convert images to RGBA mode if they're not already
    background = background.convert('RGBA')
    foreground = foreground.convert('RGBA')

    # Resize the foreground image to match the background size
    foreground = foreground.resize(background.size)

    # Merge the images by pasting the foreground onto the background
    merged_image = Image.alpha_composite(background, foreground)

    # Convert the merged image to RGB mode before saving as JPEG
    merged_image = merged_image.convert('RGB')

    # Save the merged image
    merged_image.save(output_path)

# Merge the images with specified paths
merge_images(output_image_path,output_foreground_image_path, output_image_path)
