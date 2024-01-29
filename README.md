
# Neural Style Transfer /Mixed Neural Style Transfer



## Introduction

This project involves the implementation of Neural Style Transfer (NST) and the creation of a website that provides users with an easy way to apply style transfer to their images. Initially, my plan was to implement Leon Gatys’ paper titled “A Neural Algorithm of Artistic Style” for Neural Style Transfer. However, during this period, I explored other papers related to Neural Style Transfer, one of which was another paper by Leon Gatys titled " Controlling Perceptual Factors in Neural Style Transfer " I incorporated additional features from this paper into my project website.

## Features

* **Neural Style Transfer (NST)** : Users can input a content image and a style image for style transfer.
* **Style Transfer only on Background of Image** : This option applies NST exclusively to the background of an image, useful for scenarios like styling the background of a portrait image.
* **Color Preserving Style Transfer** :NST occurs, but the color of the style image does not transfer to the content image; only paint strokes or texture are applied.
* **High Resolution Style Transfe**r : This feature not only applies style transfer but also increases the resolution of the final image.
* **Mixed Neural Style Transfer (MNST)**: Users can provide two style images—one major and one minor. The generated image showcases the style of both images. This feature also supports options like MNST on the background only, MNST with color preservation, and High-Resolution MNST.

In addition to this there is also a text to image generator that will be used style images.


## website Demo 


## Installation

### Prerequisites
1. Make sure first to create a virtual environment 
2. Then open the virtual environment 
3. Download some prequistes libraries :
```
pip install torch
pip install torchvision
pip install numpy 
pip install matplotlib
pip install skimage
pip install --upgrade diffusers transformers -q
pip install flask 
```
* If still there is any issue regarding the import of any library then install those 

Do the following in the Virtual environment:

First open the terminal :

```
git clone https://github.com/rtb-12/Neural-Style-Transfer.git
```
then 

```
cd Neural-Style-Transfer
```
Now create a models folder in this directory and paste the models file there which can be downloaded from the following link:
[https://drive.google.com/drive/folders/15OVwtPRnxwxZwKD6qPrKo6RbfsPt-lhA?usp=sharing](https://drive.google.com/drive/folders/15OVwtPRnxwxZwKD6qPrKo6RbfsPt-lhA?usp=sharing)

Here then go to > App folder > Run `main.py `

### One can Follow the following report for more understanding of the model implemented in the lab 




