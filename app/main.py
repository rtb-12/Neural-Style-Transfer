import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms
import torch.optim as optim
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.exposure import cumulative_distribution
from rembg import remove
from pathlib import Path
import tqdm
import pandas as pd
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import cv2

"""LOADING MODEL"""

vgg=models.vgg19(pretrained=True).features

for parameter in vgg.parameters():
    parameter.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)
"""Image lodaer and Unloader"""
def imageLoader(img,size):
  img_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])
  img=img_transform(img).unsqueeze(0)
  return img

def imageUnLoader(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

"""Defining Feature Map Extractor"""
def featureMapExtractor(image,model,layers):
  if layers=="style":
    layers=["1", "6", "11", "20", "29"]

  if layers=="content":
    layers= ["22"]

  if layers=="generated":
    layers=["22","1", "6", "11", "20", "29"]


  features={}
  x=image
  for name, layer in model._modules.items():
    if isinstance(layer, nn.ReLU):
        x = F.relu(x, inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        x = nn.AvgPool2d(2, 2)(x)
    else:
        x = layer(x)

    if name in layers:
        features[name] = x
  return features

"""Defining GramMatrix"""
def gramMatrix(tensor):
  b,d,h,w=tensor.size()

  tensor=tensor.view(b*d,h*w)

  gram = torch.mm(tensor, tensor.t())

  return gram

def style_transfer(content_img, style_img1, style_img2=None, alpha=1e0, beta=1e6, gamma=0, num_of_steps=500, show_iter=100):
    print("style_transfer initiated")
    size = 256  
    content = imageLoader(content_img, size).to(device)
    style1 = imageLoader(style_img1, size).to(device)
    style2 = imageLoader(style_img2, size).to(device) if style_img2 else None

    imgs_tensor = [content, style1, style2]
    content, style1, style2 = imgs_tensor

    content_fms = featureMapExtractor(content, vgg, "content")
    style_fms1 = featureMapExtractor(style1, vgg, "style")
    style_gram1 = {layer: gramMatrix(style_fms1[layer]) for layer in style_fms1}

    if style2 is not None:
        style_fms2 = featureMapExtractor(style2, vgg, "style")
        style_gram2 = {layer: gramMatrix(style_fms2[layer]) for layer in style_fms2}
    else:
        style_gram2 = style_gram1

    generated_img = content.clone().requires_grad_(True).to(device)

    style_weights = [1e3/n**2 for n in [64, 128, 256, 512, 512]]
    style_layers = ["1", "6", "11", "20", "29"]

    optimizer = optim.Adam(params=[generated_img], lr=0.003)

    step = 0
    while step <= num_of_steps:
        generated_img_features = featureMapExtractor(generated_img, vgg, "generated")

        content_loss = torch.mean((generated_img_features["22"] - content_fms["22"])**2)

        style_loss1 = 0
        i = 0
        for layer in style_layers:
            generated_img_feature = generated_img_features[layer]
            generated_img_gram = gramMatrix(generated_img_feature)
            _, d, h, w = generated_img_feature.shape
            style_img_gram1 = style_gram1[layer]
            layer_style_loss = style_weights[i] * torch.mean((generated_img_gram - style_img_gram1)**2)
            style_loss1 += layer_style_loss / (d * h * w)
            i += 1

        style_loss2 = 0
        i = 0
        for layer in style_layers:
            generated_img_feature = generated_img_features[layer]
            generated_img_gram = gramMatrix(generated_img_feature)
            _, d, h, w = generated_img_feature.shape
            style_img_gram2 = style_gram2[layer]
            layer_style_loss = style_weights[i] * torch.mean((generated_img_gram - style_img_gram2)**2)
            style_loss2 += layer_style_loss / (d * h * w)
            i += 1

        total_loss = alpha * content_loss + beta * (gamma*style_loss1 + (1-gamma)*style_loss2)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % show_iter == 0:
            print('Step {}: Total loss: {:.4f}'.format(step, total_loss.item()))
        

        step += 1
    

    return generated_img
#extract background image
def extract_background_foreground(input_path):
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
    
    background.save(os.path.join(static_dir,"output_background_path.png"))
    subject.save(os.path.join(static_dir,"output_foreground_path.png"))
    print("Output images saved.")
def remove_background_advanced(image):
    # Convert the image to RGBA mode
    image = image.convert("RGBA")

    # Convert PIL image to numpy array
    data = np.array(image)

    # Set threshold for black pixels
    lower_black = np.array([0, 0, 0, 255])
    upper_black = np.array([20, 20, 20, 255])

    # Create a mask to identify black pixels
    mask = np.all(data[:, :, :3] <= upper_black[:3], axis=-1) & np.all(data[:, :, :3] >= lower_black[:3], axis=-1)

    # Apply erosion and dilation
    mask = binary_erosion(mask, structure=np.ones((5, 5)))
    mask = binary_dilation(mask, structure=np.ones((5, 5)))

    # Apply the mask to remove black background
    data[mask] = [0, 0, 0, 0]  # Set black pixels to transparent

    # Convert the numpy array back to a PIL image
    image = Image.fromarray(data, 'RGBA')
    return image
#merge background  and foreground
def merge_images(background_path, foreground_path):
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
    merged_image.save(os.path.join(static_dir,"final_image.png"))

#color histogram mathcing
def getCDF(image):
    cdf, bins = cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0] * bins[0])
    cdf = np.append(cdf, [1] * (255 - bins[-1]))
    return cdf

def histMatch(cdfInput, cdfTemplate, imageInput):
    pixelValues = np.arange(256)
    new_pixels = np.interp(cdfInput, cdfTemplate, pixelValues)
    imageMatch = (np.reshape(new_pixels[imageInput.ravel()], imageInput.shape)).astype(np.uint8)
    return imageMatch

def histogram_matching(content_image, style_image):
    # Convert PIL Image to NumPy array
    content_array = np.array(content_image)
    style_array = np.array(style_image)

    # create a matrix for the result
    result_image = np.zeros_like(content_array).astype(np.uint8)

    # cdf and histogram matching
    for channel in range(3):
        cdf_content = getCDF(content_array[:, :, channel])
        cdf_style = getCDF(style_array[:, :, channel])
        result_image[:, :, channel] = histMatch(cdf_content, cdf_style, content_array[:, :, channel])

    return result_image

#high res transfer:
def high_res_transfer(content_img, style_img1, style_img2=None, alpha=1e0, beta=1e6, gamma=0, num_of_steps=25, show_iter=5):
    print("high_res_transfer initiated")
    image_hr = 1024

    # Assuming device is defined elsewhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content = imageLoader(content_img, image_hr).to(device)
    style1 = imageLoader(style_img1, image_hr).to(device)
    style2 = imageLoader(style_img2, image_hr).to(device) if style_img2 else None

    imgs_tensor = [content, style1, style2]
    content, style1, style2 = imgs_tensor

    content_fms = featureMapExtractor(content, vgg, "content")
    style_fms1 = featureMapExtractor(style1, vgg, "style")
    style_gram1 = {layer: gramMatrix(style_fms1[layer]) for layer in style_fms1}

    if style2 is not None:
        style_fms2 = featureMapExtractor(style2, vgg, "style")
        style_gram2 = {layer: gramMatrix(style_fms2[layer]) for layer in style_fms2}
    else:
        style_gram2 = style_gram1

    generated_img_hr = content.clone().requires_grad_(True).to(device)

    style_weights = [1e12 / n**2 for n in [64, 128, 256, 512, 512]]
    style_layers = ["1", "6", "11", "20", "29"]

    optimizer = optim.Adam(params=[generated_img_hr], lr=0.003)

    step = 0
    while step <= num_of_steps:
        generated_img_hr_features = featureMapExtractor(generated_img_hr, vgg, "generated")

        content_loss = torch.mean((generated_img_hr_features["22"] - content_fms["22"])**2)

        style_loss1 = 0
        style_loss2 = 0

        i = 0
        for layer in style_layers:
            generated_img_hr_feature = generated_img_hr_features[layer]
            generated_img_hr_gram = gramMatrix(generated_img_hr_feature)
            _, d, h, w = generated_img_hr_feature.shape
            style_img_gram1 = style_gram1[layer]
            layer_style_loss1 = style_weights[i] * torch.mean((generated_img_hr_gram - style_img_gram1)**2)
            style_loss1 += layer_style_loss1 / (d * h * w)
            i += 1

        i = 0
        for layer in style_layers:
            generated_img_hr_feature = generated_img_hr_features[layer]
            generated_img_hr_gram = gramMatrix(generated_img_hr_feature)
            _, d, h, w = generated_img_hr_feature.shape
            style_img_gram2 = style_gram2[layer]
            layer_style_loss2 = style_weights[i] * torch.mean((generated_img_hr_gram - style_img_gram2)**2)
            style_loss2 += layer_style_loss2 / (d * h * w)
            i += 1

        total_loss = alpha * content_loss + beta * (gamma * style_loss1 + (1 - gamma) * style_loss2)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % show_iter == 0:
            print('Step {}: Total loss: {:.4f}'.format(step, total_loss.item()))

        step += 1

    generated_out_img_hr = imageUnLoader(generated_img_hr)

    # Ensure the data type is uint8
    generated_out_img_hr = (generated_out_img_hr * 255).astype(np.uint8)

    # Create the PIL Image object
    generated_out_img_hr_pil = Image.fromarray(generated_out_img_hr)

    # Save the image as a file
    generated_out_img_hr_pil.save(os.path.join(static_dir,"generated_image_hr.png"))

"""Text to Image generator"""
# class CFG: 
#   seed = 42
#   if torch.cuda.is_available():
#     device = "cuda"
#     generator = torch. Generator (device).manual_seed (seed)
#   else:
#     device = "cpu"
#   generator = torch. Generator (device).manual_seed (seed)
#   image_gen_steps = 1
#   image_gen_model_id = "stabilityai/stable-diffusion-2"
#   image_gen_size = (250,250)
#   image_gen_guidance_scale = 9
#   prompt_gen_model_id = "gpt3"
#   prompt_dataset_size = 6
#   prompt_max_length = 12

# image_gen_model = StableDiffusionPipeline.from_pretrained(
#     CFG.image_gen_model_id, torch_dtype=torch.float32,
#     revision="fp16", use_auth_token='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', guidance_scale=9
# )
# image_gen_model = image_gen_model.to(CFG.device)

# def generate_style_image(prompt):
#     image = image_gen_model(
#         prompt, num_inference_steps=CFG.image_gen_steps,
#         generator=CFG.generator,
#         guidance_scale=CFG.image_gen_guidance_scale
#     ).images[0]

#     image = image.resize(CFG.image_gen_size)
#     image.save(os.path.join(static_dir,"generated_style_img.jpg"))




#flask app
from flask import Flask, render_template, request, redirect, url_for,send_file,jsonify

app = Flask(__name__)
static_dir = os.path.join(app.root_path, 'static')

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'content_image' not in request.files or 'style_image1' not in request.files:
            return redirect(request.url)

        content_image = request.files['content_image']
        style_image1 = request.files['style_image1']
        
        content_image = Image.open(content_image).convert('RGB')
        style_image1 = Image.open(style_image1).convert('RGB')
       
        generated_image = style_transfer(content_image, style_image1, style_img2=None, alpha=1e0, beta=1e6, gamma=0.25, num_of_steps=200, show_iter=50)
        generated_out_img=imageUnLoader(generated_image)
        
        # Ensure the data type is uint8
        generated_out_img = (generated_out_img * 255).astype(np.uint8)

        # Create the PIL Image object
        generated_out_img_pil = Image.fromarray(generated_out_img)

        generated_out_img_pil.save(os.path.join(static_dir,"generated.jpg"))  # Save generated image
        print("got requested image")
        return redirect(url_for('get_generated_image'))

    return render_template('index.html')

@app.route('/background_style_transfer', methods=['POST'])

def background_style_transfer():
    if 'content_image' not in request.files or 'style_image1' not in request.files:
        return redirect(request.url)

    content_file = request.files['content_image']
    style_file = request.files['style_image1']

    # Save uploaded images to temporary files
    content_temp_path = os.path.join(static_dir,"content_temp.jpg")
    style_temp_path = os.path.join(static_dir,"style_temp.jpg")
    background_img=os.path.join(static_dir,"output_background_path.png")

    content_file.save(content_temp_path)
    style_file.save(style_temp_path)
    # background_img.save(background_img)

    # Load uploaded images as PIL Image objects
    content_image = Image.open(content_temp_path).convert('RGB')
    style_image1 = Image.open(style_temp_path).convert('RGB')
    background_img = Image.open(background_img)

    # Perform background extraction and style transfer
    extract_background_foreground(content_temp_path)
    
    background_style=style_transfer(background_img,style_image1, style_img2=None, alpha=1e0, beta=1e6, gamma=0, num_of_steps=200, show_iter=50)
    background_style=imageUnLoader(background_style)
    # Ensure the data type is uint8
    background_style = (background_style * 255).astype(np.uint8)
    background_style_pil=Image.fromarray(background_style)
    background_style_pil.save(os.path.join(static_dir,"Styled_Background_image_path.jpg"))

    # Merge background and foreground images
    merge_images( os.path.join(static_dir,"Styled_Background_image_path.jpg"),  os.path.join(static_dir,"output_foreground_path.png"))
    print("got requested imageB")
    return redirect(url_for('get_generated_background_image'))


@app.route("/style_transfer_color_preserve", methods=['POST'])
def style_transfer_color_preserve():
    if 'content_image' not in request.files or 'style_image1' not in request.files:
        return redirect(request.url)

    content_file = request.files['content_image']
    style_file1 = request.files['style_image1']

    # Load uploaded images as PIL Image objects
    content_image = Image.open(content_file).convert('RGB')
    style_image1 = Image.open(style_file1).convert('RGB')

    # Perform color-preserving style transfer
    generated_style_color_preserve =histogram_matching(content_image, style_image1)
    generated_style_color_preserve_pil = Image.fromarray(generated_style_color_preserve)
    # generated_style_color_preserve_pil.save(os.path.join(static_dir,"histogramMatch_style_img.jpeg"))
    
    style_transfer_color_preserve=style_transfer(content_image,generated_style_color_preserve_pil, style_img2=None, alpha=1e0, beta=1e6, gamma=0, num_of_steps=200, show_iter=50)
    
    style_transfer_color_preserve=imageUnLoader(style_transfer_color_preserve)
    # Ensure the data type is uint8
    style_transfer_color_preserve=(style_transfer_color_preserve * 255).astype(np.uint8)
    # Create the PIL Image object
    style_transfer_color_preserve_pil=Image.fromarray(style_transfer_color_preserve)
    style_transfer_color_preserve_pil.save(os.path.join(static_dir,"style_transfer_color_preserve.jpg"))  # Save generated image
    print("got requested image")
    return redirect(url_for('get_generated_image_color_preserve'))



@app.route("/high_resolution_style_transfer", methods=['POST'])
def high_resolution_style_transfer():
        print("high resolution style transfer initiated")
        if 'content_image' not in request.files or 'style_image1' not in request.files:
            return redirect(request.url)

        content_image = request.files['content_image']
        style_image1 = request.files['style_image1']


        content_image = Image.open(content_image).convert('RGB')
        style_image1 = Image.open(style_image1).convert('RGB')

        generated_image = style_transfer(content_image, style_image1, style_img2=None, alpha=1e0, beta=1e6, gamma=0, num_of_steps=200, show_iter=50)
        generated_out_img=imageUnLoader(generated_image)
        
        # Ensure the data type is uint8
        generated_out_img = (generated_out_img * 255).astype(np.uint8)

        # Create the PIL Image object
        generated_out_img_pil = Image.fromarray(generated_out_img)

        generated_out_img_pil.save(os.path.join(static_dir,"generated.jpg"))

        content_image = Image.open(os.path.join(static_dir,"generated.jpg")).convert('RGB')
        hi_res_img=high_res_transfer(content_image, style_image1, style_img2=None, alpha=1e0, beta=1e6, gamma=0, num_of_steps=20, show_iter=5)
        return redirect(url_for('get_generated_image_high_res'))



@app.route("/second_page", methods=['GET', 'POST'])
def second_page():
    if request.method == 'POST':
        if 'content_image' not in request.files or 'style_image1' not in request.files or 'style_image2' not in request.files:
            return redirect(request.url)

        content_image = request.files['content_image']
        style_image1 = request.files['style_image1']
        style_image2 = request.files['style_image2']

        content_image = Image.open(content_image).convert('RGB')
        style_image1 = Image.open(style_image1).convert('RGB')
        style_image2 = Image.open(style_image2).convert('RGB')

        # Perform style transfer with two style images for the second page
        generated_image = style_transfer(content_image, style_image1,style_image2, alpha=1e0, beta=1e6, gamma=0.25, num_of_steps=200, show_iter=50)
        generated_out_img = imageUnLoader(generated_image)

        # Ensure the data type is uint8
        generated_out_img = (generated_out_img * 255).astype(np.uint8)

        # Create the PIL Image object
        generated_out_img_pil = Image.fromarray(generated_out_img)

        generated_out_img_pil.save(os.path.join(static_dir, "generated_image2.jpg"))  # Save generated image
        return redirect(url_for('get_generated_image2'))

# Change the endpoint function name from background_style_transfer to background_style_transfer2

@app.route('/background_style_transfer2', methods=['POST'])
def background_style_transfer2():
    if 'content_image' not in request.files or 'style_image1' not in request.files or 'style_image2' not in request.files:
        return redirect(request.url)

    content_image = request.files['content_image']
    style_image1 = request.files['style_image1']
    style_image2 = request.files['style_image2']

    content_image = Image.open(content_image).convert('RGB')
    style_image1 = Image.open(style_image1).convert('RGB')
    style_image2 = Image.open(style_image2).convert('RGB')

    # Save uploaded images to temporary files
    content_temp_path = os.path.join(static_dir, "content_temp.jpg")
    background_img = os.path.join(static_dir, "output_background_path.png")
    background_img = Image.open(background_img)
    content_image.save(content_temp_path)
    # background_img_path.save(background_img_path)

    # Load uploaded images as PIL Image objects
    content_image = Image.open(content_temp_path).convert('RGB')
    # background_img = Image.open(background_img_path)

    # Perform background extraction and style transfer
    extract_background_foreground(content_temp_path)

    background_style = style_transfer(background_img, style_image1, style_image2, alpha=1e0, beta=1e6, gamma=0.25, num_of_steps=200, show_iter=50)
    background_style = imageUnLoader(background_style)

    # Ensure the data type is uint8
    background_style = (background_style * 255).astype(np.uint8)
    background_style_pil = Image.fromarray(background_style)
    background_style_pil.save(os.path.join(static_dir, "Styled_Background_image_path2.jpg"))

    # Merge background and foreground images
    merge_images(os.path.join(static_dir, "Styled_Background_image_path2.jpg"), os.path.join(static_dir, "output_foreground_path.png"))
    print("got requested imageB2")
    return redirect(url_for('get_generated_background_image2'))


@app.route("/style_transfer_color_preserve2", methods=['POST'])
def style_transfer_color_preserve2():
    if 'content_image' not in request.files or 'style_image1' not in request.files or 'style_image2' not in request.files:
        return redirect(request.url)

    content_file = request.files['content_image']
    style_file1 = request.files['style_image1']
    style_file2 = request.files['style_image2']

    # Load uploaded images as PIL Image objects
    content_image = Image.open(content_file).convert('RGB')
    style_image1 = Image.open(style_file1).convert('RGB')
    style_image2 = Image.open(style_file2).convert('RGB')

    # Perform color-preserving style transfer
    generated_style_color_preserve1 = histogram_matching(content_image, style_image1)
    generated_style_color_preserve2 = histogram_matching(content_image, style_image2)
    generated_style_color_preserve_pil1 = Image.fromarray(generated_style_color_preserve1)
    generated_style_color_preserve_pil2 = Image.fromarray(generated_style_color_preserve2)

    style_transfer_color_preserve = style_transfer(content_image, generated_style_color_preserve_pil1, generated_style_color_preserve_pil2, alpha=1e0, beta=1e6, gamma=0.25, num_of_steps=200, show_iter=50)
    
    style_transfer_color_preserve = imageUnLoader(style_transfer_color_preserve)
    
    # Ensure the data type is uint8
    style_transfer_color_preserve = (style_transfer_color_preserve * 255).astype(np.uint8)
    
    # Create the PIL Image object
    style_transfer_color_preserve_pil = Image.fromarray(style_transfer_color_preserve)
    style_transfer_color_preserve_pil.save(os.path.join(static_dir, "style_transfer_color_preserve2.jpg"))  # Save generated image
    print("got requested image")
    return redirect(url_for('get_generated_image_color_preserve2'))

@app.route("/high_resolution_style_transfer2", methods=['POST'])
def high_resolution_style_transfer2():
        print("high resolution style transfer initiated")
        if 'content_image' not in request.files or 'style_image1' not in request.files or 'style_image2' not in request.files:
           return redirect(request.url)

        content_image = request.files['content_image']
        style_image1 = request.files['style_image1']
        style_image2 = request.files['style_image2']

        # Load uploaded images as PIL Image objects
        content_image = Image.open(content_file).convert('RGB')
        style_image1 = Image.open(style_file1).convert('RGB')
        style_image2 = Image.open(style_file2).convert('RGB')

        generated_image = style_transfer(content_image, style_image1, style_img2, alpha=1e0, beta=1e6, gamma=0.25, num_of_steps=200, show_iter=50)
        generated_out_img=imageUnLoader(generated_image)
        
        # Ensure the data type is uint8
        generated_out_img = (generated_out_img * 255).astype(np.uint8)

        # Create the PIL Image object
        generated_out_img_pil = Image.fromarray(generated_out_img)

        generated_out_img_pil.save(os.path.join(static_dir,"generated2.jpg"))

        content_image = Image.open(os.path.join(static_dir,"generated2.jpg")).convert('RGB')
        hi_res_img=high_res_transfer(content_image, style_image1, style_img2, alpha=1e0, beta=1e6, gamma=0.25, num_of_steps=20, show_iter=5)
        return redirect(url_for('get_generated_image_high_res2'))


# @app.route('/generate_image', methods=['POST'])
# def generate_image():
#     try:
#         prompt = request.form.get('prompt')
#         print(prompt)
#         img = generate_style_image(prompt)
#         image_path = os.path.join(static_dir, "generated_style_img.jpg")
#         print("Image path:", image_path)
#         return redirect(url_for('get_generated_style_image'))
#     except Exception as e:
#         print(f"Error in generate_image: {str(e)}")


# @app.route('/generated_style_image')
# def get_generated_style_image():
#     return send_file(os.path.join(static_dir, "generated_style_img.jpg"), mimetype='image/jpg')

@app.route('/generated_image2')
def get_generated_image2():
    return send_file(os.path.join(static_dir, "generated_image2.jpg"), mimetype='image/jpg')


@app.route('/generated_image_high_res')
def get_generated_image_high_res():
    return send_file(os.path.join(static_dir, "generated_image_hr.png"), mimetype='image/png')

@app.route('/generated_image_high_res2')
def get_generated_image_high_res2():
    return send_file(os.path.join(static_dir, "generated_image_hr.png"), mimetype='image/png')

@app.route('/generated_image_color_preserve')
def get_generated_image_color_preserve():
    return send_file(os.path.join(static_dir, "style_transfer_color_preserve.jpg"), mimetype='image/jpg') 
  
@app.route('/generated_image_color_preserve2')
def get_generated_image_color_preserve2():
    return send_file(os.path.join(static_dir, "style_transfer_color_preserve2.jpg"), mimetype='image/jpg')


@app.route('/generated_background_image')
def get_generated_background_image():
    return send_file( os.path.join(static_dir,"final_image.png"), mimetype='image/jpg')

@app.route('/generated_image')
def get_generated_image():
    return send_file(os.path.join(static_dir,"generated.jpg"), mimetype='image/jpg')

@app.route('/generated_background_image2')
def get_generated_background_image2():
    return send_file(os.path.join(static_dir, "final_image.jpg"), mimetype='image/jpg')
    # return redirect(url_for('get_generated_background_image2'))

if __name__ == "__main__":
    app.run(debug=True)



