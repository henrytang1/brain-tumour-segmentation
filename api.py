import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use("ggplot")
matplotlib.use('agg')

import cv2
from tqdm.notebook import tqdm
from glob import glob
from itertools import chain

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F

from PIL import Image

from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
upload_folder = "./static"
device = "cpu"
model = None
path = "./model_state_dict.pt"
data_transforms = None

def conv_layer(input_channels, output_channels):     #This is a helper function to create the convolutional blocks
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
    )
    return conv



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_1 = conv_layer(3, 64)
        self.down_2 = conv_layer(64, 128)
        self.down_3 = conv_layer(128, 256)
        self.down_4 = conv_layer(256, 512)
        self.down_5 = conv_layer(512, 1024)
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024, 512)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = conv_layer(512, 256)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(256, 128)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128, 64)
        
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)
        self.output_activation = nn.Sigmoid()
                
    def forward(self, img):     #The print statements can be used to visualize the input and output sizes for debugging
        x1 = self.down_1(img)
        #print(x1.size())
        x2 = self.max_pool(x1)
        #print(x2.size())
        x3 = self.down_2(x2)
        #print(x3.size())
        x4 = self.max_pool(x3)
        #print(x4.size())
        x5 = self.down_3(x4)
        #print(x5.size())
        x6 = self.max_pool(x5)
        #print(x6.size())
        x7 = self.down_4(x6)
        #print(x7.size())
        x8 = self.max_pool(x7)
        #print(x8.size())
        x9 = self.down_5(x8)
        #print(x9.size())
        
        x = self.up_1(x9)
        #print(x.size())
        x = self.up_conv_1(torch.cat([x, x7], 1))
        #print(x.size())
        x = self.up_2(x)
        #print(x.size())
        x = self.up_conv_2(torch.cat([x, x5], 1))
        #print(x.size())
        x = self.up_3(x)
        #print(x.size())
        x = self.up_conv_3(torch.cat([x, x3], 1))
        #print(x.size())
        x = self.up_4(x)
        #print(x.size())
        x = self.up_conv_4(torch.cat([x, x1], 1))
        #print(x.size())
        
        x = self.output(x)
        x = self.output_activation(x)
        #print(x.size())
        
        return x


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


def process_image(data_transforms, path_name, image_name, filemodel):
    with torch.no_grad():
        img = image_loader(data_transforms, path_name)
        pred = model(img)
        
        # plt.figure(figsize=(3,2))
        plt.subplot(1,2,1)
        plt.imshow(np.squeeze(img.cpu().numpy()).transpose(1,2,0))
        plt.title('Original Image')
        # plt.subplot(1,3,2)
        # plt.imshow((mask.cpu().numpy()).transpose(1,2,0).squeeze(axis=2), alpha=0.5)
        # plt.title('Original Mask')
        plt.subplot(1,2,2)
        plt.imshow(np.squeeze(pred.cpu()) > .5)
        plt.title('Tumour Prediction')
        
        
        plt.savefig("%s/%s-NEW.png" % (upload_folder, image_name), bbox_inches = "tight")
        # plt.show()



# @app.route("/", methods=["GET", "POST"])
# def upload_predict():
#     if request.method == "POST":
#         image_file = request.files["image"]
#         if image_file:
#             image_location = os.path.join(
#                 upload_folder,
#                 image_file.filename
#             )
#             image_file.save(image_location)

#             image_name = os.path.basename(image_location)
#             image_name = image_name.split('.')[0]
#             print(image_name)

#             # print(image_file)
#             # print(image_location)
#             # pred = 1
#             # pred = predict(image_location, MODEL)[0]
#             process_image(data_transforms, image_location, image_name, model)

#             # print("%s/%s-NEW.png" % (upload_folder, image_name))

#             print("%s-NEW.png" % image_name)
#             # return render_template("index.html", image_loc = ("%s-NEW.png" % image_name))
#             # print(image_file.filename)
#             return render_template("index.html", image_loc = image_file.filename)

#     return render_template("index.html", image_loc=None)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)

            image_name = os.path.basename(image_location)
            image_name = image_name.split('.')[0]
            print(image_name)
            print(image_location)

            process_image(data_transforms, image_location, image_name, model)

            # pred = 1
            # pred = predict(image_location, MODEL)[0]
            return render_template("index.html", image_loc = ("%s-NEW.png" % image_name))
            
    return render_template("index.html", prediction=0, image_loc=None)



if __name__ == "__main__":
    #Initialize the model and optimizer
    model = UNet().to(device)

    # Load a preexisting set of weights if continuting training
    # model.load_state_dict(torch.load(PATH), map_location = torch.device('cpu'))

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    #Test the model with some samples from the test dataset 
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    # img_path = "./static/thing2.tiff"
    # process_image(data_transforms, img_path, "hi", model)

    # img = image_loader(data_transforms, img_path)
    # pred = model(img)

    # # plt.figure(figsize=(3,2))
    # plt.subplot(1,2,1)
    # plt.imshow(np.squeeze(img.cpu().numpy()).transpose(1,2,0))
    # plt.title('Original Image')
    # # plt.subplot(1,3,2)
    # # plt.imshow((mask.cpu().numpy()).transpose(1,2,0).squeeze(axis=2), alpha=0.5)
    # # plt.title('Original Mask')
    # plt.subplot(1,2,2)
    # plt.imshow(np.squeeze(pred.cpu()) > .5)
    # plt.title('Prediction')
    # plt.show()
    
    # process_image()

    app.run(host="0.0.0.0", port=12000, debug=True)
