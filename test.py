import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Paths for image directory and model
IMDIR = sys.argv[1]
MODELDIR = sys.argv[2]
OUTDIR = sys.argv[3]
OUTPATH = os.path.join(OUTDIR, os.path.split(IMDIR)[-1])

# Load the model for testing
model = torch.load(MODELDIR)
model.eval()

# Class labels for prediction
class_names=['birds', 'cats']

# Configure plots
fig = plt.figure()

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform prediction and plot results
with torch.no_grad():
    img = Image.open(IMDIR).convert('RGB')
    inputs = preprocess(img).unsqueeze(0).to(device)
    outputs = model(inputs)
    probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
    plt.title("Birds: {0:.1f}%\n Cats: {1:.1f}%".format(probs[0]*100.0, probs[1]*100.0))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(OUTPATH)