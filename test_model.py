from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# helper imports
from model_helper import load_checkpoint

# image imports
from skimage import io, transform
from PIL import Image

# general imports
import re
import os, glob
import time
import datetime
import pdb
from shutil import copyfile
from shutil import rmtree
from pathlib import Path

# data science imports
import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import statistics
    
import modified_densenet
import modified_alexnet

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()

def init_model(MODEL_NAME, N_LABELS):
    # Check model used
    model = (models.densenet121(pretrained=True) if MODEL_NAME == 'densenet' 
            else modified_densenet.densenet121(type=MODEL_NAME, pretrained=True) 
                if MODEL_NAME == 'va-densenet' 
                or MODEL_NAME == 'reva-densenet' 
                or MODEL_NAME == 'fp-densenet' 
                or MODEL_NAME == 'start-densenet' 
                or MODEL_NAME == 'every-densenet' 
                or MODEL_NAME == 'sedensenet'
                or MODEL_NAME == 'triplelossdensenet'
            else models.alexnet(pretrained=True) if MODEL_NAME == 'alexnet'
            else modified_alexnet.alexnet(type=MODEL_NAME, pretrained=True) 
                if MODEL_NAME == 'va-alexnet'
                or MODEL_NAME == 'reva-alexnet'
                or MODEL_NAME == 'fp-alexnet'
                or MODEL_NAME == 'start-alexnet'
            else models.resnet152(pretrained=True)if MODEL_NAME == 'resnet'
            else models.vgg16(pretrained=True)if MODEL_NAME == 'VGG'
            else models.vgg16_bn(pretrained=True)if MODEL_NAME == 'VGG_Bn'
            else '')

    # get num_ftrs based on model name
    num_ftrs = (model.classifier.in_features 
                    if MODEL_NAME == 'densenet' 
                    or MODEL_NAME == 'va-densenet' 
                    or MODEL_NAME == 'reva-densenet' 
                    or MODEL_NAME == 'fp-densenet' 
                    or MODEL_NAME == 'start-densenet' 
                    or MODEL_NAME == 'every-densenet' 
                    or MODEL_NAME == 'sedensenet'
                    or MODEL_NAME == 'triplelossdensenet'
                else model.classifier[6].in_features 
                    if MODEL_NAME == 'alexnet' 
                    or MODEL_NAME == 'va-alexnet' 
                    or MODEL_NAME == 'reva-alexnet' 
                    or MODEL_NAME == 'fp-alexnet' 
                    or MODEL_NAME == 'start-alexnet' 
                    or MODEL_NAME == 'VGG' 
                    or MODEL_NAME == 'VGG_Bn'
                else model.fc.in_features 
                    if MODEL_NAME == 'resnet'
                else model.fc3.in_features 
                    if MODEL_NAME == 'small_va'
                else '')

    # change classifier class to N_LABELS
    if (MODEL_NAME == 'densenet' or MODEL_NAME == 'va-densenet' or MODEL_NAME == 'reva-densenet' or MODEL_NAME == 'fp-densenet' or MODEL_NAME == 'start-densenet' or MODEL_NAME == 'every-densenet' or MODEL_NAME == 'sedensenet' or MODEL_NAME == 'triplelossdensenet'):
        model.classifier = nn.Linear(num_ftrs, N_LABELS)
    elif (MODEL_NAME == 'alexnet' or MODEL_NAME == 'va-alexnet' or MODEL_NAME == 'va-alexnet' or MODEL_NAME == 'reva-alexnet' or MODEL_NAME == 'fp-alexnet' or MODEL_NAME == 'start-alexnet' or MODEL_NAME == 'VGG' or MODEL_NAME == 'VGG_Bn'):
        model.classifier[6] = nn.Linear(num_ftrs, N_LABELS)
    elif (MODEL_NAME == 'resnet'):
        model.fc = nn.Linear(num_ftrs, N_LABELS)
    else:
        raise ValueError("Error model name")

    return model


def test_cnn(MODEL_NAME, MODEL_NAME_TARGET, BATCH_SIZE, N_LABELS, PATH_TO_IMAGES, DEBUG_MODE, CHECKPOINT_PATH, CHECKPOINT_PATH_TARGET):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        MODEL_NAME: model name
        MODEL_NAME_TARGET: the other model name
        BATCH_SIZE: number of batch data per training
        N_LABELS: number of class labels
        PATH_TO_IMAGES: path to NIH images
        DEBUG_MODE: if true then no log will be created
        CHECKPOINT_PATH: load checkpoint path
        CHECKPOINT_PATH_TARGET: load the other checkpoint path
    Returns:
        # preds: torchvision model predictions on test fold with ground truth for comparison
        # aucs: AUCs for each train,test tuple
    """

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # create train/val dataloaders
    transformed_datasets = {x: datasets.ImageFolder(os.path.join(PATH_TO_IMAGES, x), data_transforms[x]) for x in ['val']}

    dataloaders = {x: torch.utils.data.DataLoader(transformed_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['val']}

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    model = init_model(MODEL_NAME, N_LABELS)
    model = load_checkpoint(model, CHECKPOINT_PATH)
    model = model.cuda()

    if (CHECKPOINT_PATH_TARGET):
        model_target = init_model(MODEL_NAME_TARGET, N_LABELS)
        model_target = load_checkpoint(model_target, CHECKPOINT_PATH_TARGET)
        model_target = model_target.cuda()

    loading_bar = ''
    dataloaders_length = len(dataloaders['val'])
    for i in range(dataloaders_length):
        loading_bar += '-'

    model_labels = []
    model_pred = []
    model_target_pred = []

    model_pred_bin = []
    model_target_pred_bin = []

    for phase in ['val']:
        model.eval()
        model_target.eval()
        for data in dataloaders[phase]:
            loading_bar = f'={loading_bar}'
            loading_bar = loading_bar[:dataloaders_length]
            print(f'Testing: {loading_bar}', end='\r')
            inputs, labels = data

            labels = labels.cpu().data.numpy()

            if phase == 'val':
                inputs = inputs.cuda()
                model_labels.extend(labels)
                try:
                    outputs = model(inputs, labels)
                except:
                    outputs = model(inputs)
                outputs_pred = torch.max(outputs, dim=1)[1].cpu().data.numpy()
                model_pred.extend(outputs_pred)
                if (CHECKPOINT_PATH_TARGET):
                    try:
                        outputs_target = model_target(inputs, labels)
                    except:
                        outputs_target = model_target(inputs)
                    outputs_target_pred = torch.max(outputs_target, dim=1)[1].cpu().data.numpy()
                    model_target_pred.extend(outputs_target_pred)

    print('')
    print(confusion_matrix(model_labels, model_pred, labels=[1, 0, 2, 4, 3, 5, 6]))
    if (CHECKPOINT_PATH_TARGET):
        print(confusion_matrix(model_labels, model_target_pred, labels=[1, 0, 2, 4, 3, 5, 6]))
    print('============================================')
    print('precision: ', precision_score(model_labels, model_pred, average='macro'))
    print('recall:    ', recall_score(model_labels, model_pred, average='macro'))
    print('f1:        ', f1_score(model_labels, model_pred, average='macro'))
    if (CHECKPOINT_PATH_TARGET):
        print('============================================')
        print('precision: ', precision_score(model_labels, model_target_pred, average='macro'))
        print('recall:    ', recall_score(model_labels, model_target_pred, average='macro'))
        print('f1:        ', f1_score(model_labels, model_target_pred, average='macro'))
        print('============================================')

    for i, _ in enumerate(model_labels):
        model_pred_bin.append(1 if model_labels[i] == model_pred[i] else 0)
        if (CHECKPOINT_PATH_TARGET):
            model_target_pred_bin.append(1 if model_labels[i] == model_target_pred[i] else 0)

    print(accuracy_score(model_labels, model_pred))

    if (CHECKPOINT_PATH_TARGET):
        print(accuracy_score(model_labels, model_target_pred))
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, _ in enumerate(model_pred_bin):
            if model_pred_bin[i] == model_target_pred_bin[i]:
                if model_pred_bin[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if model_pred_bin[i] == 0:
                    fp += 1
                else:
                    fn += 1

        print(f"True positive = {tp}")
        print(f"False positive = {fp}")
        print(f"False negative = {fn}")
        print(f"True negative = {tn}")

        print("Finish testing")
