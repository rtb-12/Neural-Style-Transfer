
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
![Screenshot 2024-01-28 170621](https://github.com/rtb-12/Neural-Style-Transfer/assets/147048280/68e5f0e4-6442-43da-a11a-4e99649c857b)
![Screenshot 2024-01-29 211154](https://github.com/rtb-12/Neural-Style-Transfer/assets/147048280/bb1c0b15-c640-473c-ad2c-aa25d53cdc68)
![Screenshot 2024-01-29 205410](https://github.com/rtb-12/Neural-Style-Transfer/assets/147048280/e95cc88b-7ca8-40a2-aaa6-f3a1d0c69fe5)


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
Now create a `Models` folder in this directory and paste the models file there which can be downloaded from the following link:
[https://drive.google.com/drive/folders/15OVwtPRnxwxZwKD6qPrKo6RbfsPt-lhA?usp=sharing](https://drive.google.com/drive/folders/15OVwtPRnxwxZwKD6qPrKo6RbfsPt-lhA?usp=sharing)

NOTE : For Text to image generation one requires a hugging face auth token which can be generated on hugging face webiste and then paste it the code where auth token is written (Currently on auth token is already there for testing purposes and will be soon removed)

Here then go to > App folder > Run `main.py `

## Results 
* **Neural Style Transfer:**
![generated_image (2) (1)](https://hackmd.io/_uploads/HkOqzAQcT.png)
* **Color Preserving Style Transfer :**
  
![generated_image (3)](https://hackmd.io/_uploads/Bk81XCQ9p.png)

* **Background only Style Transfer :**
![CONTENT                        STYLE](https://hackmd.io/_uploads/BJc6KZN5p.jpg)

Generated Image:

![final_image](https://hackmd.io/_uploads/HkPNQWN5a.png)

* **High Resolution Style Transfer :**
![generated_image_hr (2)](https://github.com/rtb-12/Neural-Style-Transfer/assets/147048280/156849b5-d70e-49d2-b60b-a9318347e532)
* **Mixed Style Transfer :**
![CONTENT            STYLE1             STYLE2](https://hackmd.io/_uploads/ryCDXb456.jpg)
Generated Image:

![MNST](https://hackmd.io/_uploads/HJDAKk496.png)


### NOTE:
Currently the number of steps for image generation from prompt is kept :10 
for style transfer :200(size of one dimension of image :256)
and for hig res transfer :25 (size of one dimension of image:1024)
these are kept less for testing on low end laptop but can be modified in code as required based on the hardware of the device.



### One can Follow the following report for more understanding of the model implemented in the lab:
[https://hackmd.io/@nZL7vfaUTsqsyyZrlF9LXw/S1bx76756](https://hackmd.io/@nZL7vfaUTsqsyyZrlF9LXw/S1bx76756)




