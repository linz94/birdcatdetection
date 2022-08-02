import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
from utils.visualizer import plot_classes_preds
from utils.config import getConfig


if __name__ == '__main__':

    # Construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_dir', type=str, help='Root directory for training configs.')
    args = ap.parse_args()
    args = getConfig(args.config_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Applying transforms to the data
    image_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    
    # Load data from folders
    dataset = {
        'train': datasets.ImageFolder(root=args.train_dir, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=args.val_dir, transform=image_transforms['valid'])
    }
    
    # Size of train and validation data
    dataset_sizes = {
        'train':len(dataset['train']),
        'valid':len(dataset['valid'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train':data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_cpu, pin_memory=True, drop_last=True),
        'valid':data.DataLoader(dataset['valid'], batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_cpu, pin_memory=True, drop_last=True)
    }

    # Class names or target labels
    class_names = dataset['train'].classes
    print("Classes:", class_names)
    
    # Print the train and validation data sizes
    print("Training-set size:",dataset_sizes['train'],
        "\nValidation-set size:", dataset_sizes['valid'])

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a pretrained model - MobilenetV2
    print("\nLoading mobilenetv2 as feature extractor ...\n")
    model_ft = models.mobilenet_v2(pretrained=True)    

    # Freeze all the required layers (i.e except last conv block and fc layers)
    for params in list(model_ft.parameters())[0:-5]:
        params.requires_grad = False

    # Modify fc layers to match num_classes
    num_ftrs=model_ft.classifier[-1].in_features
    model_ft.classifier=nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=args.nclasses, bias=True)
        )    

    # Transfer the model to GPU
    model_ft = model_ft.to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer 
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=args.momentum)

    # Learning rate decay
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.lr_stepsize, gamma=args.lr_gamma)

    # Model training routine 
    print("\nTraining:-\n")
    def train_model(model, criterion, optimizer, scheduler, n_epochs):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # Tensorboard summary
        writer = SummaryWriter(args.logs_dir)
        
        for epoch in range(n_epochs):
            print('Epoch {}/{}'.format(epoch, n_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=2).float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # Record training loss and accuracy for each phase
                if phase == 'train':
                    writer.add_scalar('Train/Loss', epoch_loss, epoch)
                    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                    writer.flush()
                else:
                    writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                    writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                    writer.add_figure('Predictions',
                                plot_classes_preds(model, inputs, labels, class_names),
                                epoch)
                    writer.flush()

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Train the model
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                        n_epochs=args.n_epochs)
    # Save the entire model
    print("\nSaving the model...")
    torch.save(model_ft, args.checkpoints_dir)