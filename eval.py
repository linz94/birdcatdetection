import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import sys
import multiprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils.metrics import compute_all_metrics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Paths for image directory and model
EVAL_DIR = sys.argv[1]
EVAL_MODEL = sys.argv[2]

# Load the model for evaluation
model = torch.load(EVAL_MODEL)
model.eval()

# Configure batch size and nuber of cpu's
num_cpu = 6
bs = 1

# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

eval_dataset=datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader=data.DataLoader(eval_dataset, batch_size=bs, shuffle=False,
                            num_workers=num_cpu, pin_memory=True)

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes and dataset-size
num_classes = len(eval_dataset.classes)
dsize = len(eval_dataset)

# Class label names
class_names=['birds', 'cats']

# Initialize the prediction and label lists
predlist = []
lbllist = []

# Evaluate the model accuracy on the dataset

with torch.no_grad():
    
    for images, labels in eval_loader:
        images, labels = images.to(device), labels.to(device)
        labels = torch.nn.functional.one_hot(labels, num_classes=2).float()

        outputs = torch.sigmoid(model(images)).round()

        lbllist.append(labels.cpu())
        predlist.append(outputs.cpu())

    lbllist = torch.cat(lbllist, dim=0)
    predlist = torch.cat(predlist, dim=0)

# compute multi-label metrics
compute_all_metrics(lbllist.numpy(), predlist.numpy())


# compute per-class confusion matrix and metrics
plot = False
conf_mat=confusion_matrix(lbllist[:,0].numpy(), predlist[:,0].numpy())
print('Confusion Matrix Birds')
print('-'*16)
print(conf_mat,'\n')
if plot:
    df_cm = pd.DataFrame(conf_mat, ['no birds', 'birds'], ['no birds', 'birds'])
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.show()
    # plt.savefig('/cluster/home/linzhan/birdcat_detection/results/confusion_birds.png')

conf_mat=confusion_matrix(lbllist[:,1].numpy(), predlist[:,1].numpy())
print('Confusion Matrix Cats')
print('-'*16)
print(conf_mat,'\n')
if save:
    df_cm = pd.DataFrame(conf_mat, ['no cats', 'cats'], ['no cats', 'cats'])
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.show()
    # plt.savefig('/cluster/home/linzhan/birdcat_detection/results/confusion_cats.png')


print('Accuracy of class %8s : %0.2f %%'%(class_names[0], 100*accuracy_score(lbllist[:,0], predlist[:,0])))
print('Accuracy of class %8s : %0.2f %%'%(class_names[1], 100*accuracy_score(lbllist[:,1], predlist[:,1])))

print('Recall of class %8s : %0.2f %%'%(class_names[0], 100*recall_score(lbllist[:,0], predlist[:,0])))
print('Recall of class %8s : %0.2f %%'%(class_names[1], 100*recall_score(lbllist[:,1], predlist[:,1])))

print('Precision of class %8s : %0.2f %%'%(class_names[0], 100*precision_score(lbllist[:,0], predlist[:,0])))
print('Precision of class %8s : %0.2f %%'%(class_names[1], 100*precision_score(lbllist[:,1], predlist[:,1])))

print('F1 score of class %8s : %0.2f %%'%(class_names[0], 100*f1_score(lbllist[:,0], predlist[:,0])))
print('F1 score of class %8s : %0.2f %%'%(class_names[1], 100*f1_score(lbllist[:,1], predlist[:,1])))