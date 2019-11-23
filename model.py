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
from sklearn.metrics import accuracy_score
import statistics

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def create_checkpoint(model, best_loss, epoch, PRETRAINED, FREEZE, TIME, MODEL, OPTIM, LR, STEP, TRAIN_LOSS, TRAIN_ACC, VAL_LOSS, VAL_ACC):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training

        PRETRAINED: if use pretrained model
        FREEZE: if freeze base layer
        TIME: time training started (UNIX)
        MODEL: model name used
        OPTIM: optimizer name
        LR: learning rate value
        STEP: how much steps LR dropped

        TRAIN_LOSS: list of train loss
        TRAIN_ACC: list of train accuracy
        VAL_LOSS: list of val loss
        VAL_ACC: list of val accuracy
        NOTES: list of notes

    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    for filename in glob.glob(f'results/{TIME}*'):
        os.remove(filename)

    path_name = f'results/{TIME}#{MODEL}_P-{PRETRAINED}_F-{FREEZE}_{OPTIM}_LR({LR})_every-{STEP}-step_VLOSS-{VAL_LOSS[-1]}_VACC-{VAL_ACC[-1]}'

    torch.save(model.state_dict(), path_name)
    return path_name


def create_csv(DEBUG, PRETRAINED, FREEZE, TIME, MODEL, OPTIM, LR, STEP, TRAIN_LOSS, TRAIN_ACC, VAL_LOSS, VAL_ACC, NOTES, CHECKPOINT):
    """
    Create training results in csv

    Args:
        DEBUG: didn't create csv if debugging mode
        PRETRAINED: if use pretrained model
        FREEZE: if freeze base layer
        TIME: time training started
        MODEL: model name used
        OPTIM: optimizer name
        LR: learning rate value
        STEP: how much steps LR dropped

        TRAIN_LOSS: list of train loss
        TRAIN_ACC: list of train accuracy
        VAL_LOSS: list of val loss
        VAL_ACC: list of val accuracy
        NOTES: list of notes
        CHECKPOINT: checkpoint name
    """
    if not DEBUG:
        df = pd.DataFrame({
            'train_loss': TRAIN_LOSS,
            'train_acc': TRAIN_ACC,
            'val_loss': VAL_LOSS,
            'val_acc': VAL_ACC,
            'notes': NOTES
        })
        df.to_csv(f'results_csv/{TIME}#{MODEL}_CP-{CHECKPOINT}_P-{PRETRAINED}_F-{FREEZE}_{OPTIM}_LR({LR})_every-{STEP}-steps.csv')

def get_distillate_output(distillate_results, distillate_index, outputs_row):
    return torch.from_numpy(distillate_results[distillate_index][:outputs_row, :]).float().cuda()

def train_model(
        model,
        criterion,
        optimizer,
        optim_name,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        scheduler,
        debug_mode,
        pretrained,
        freeze,
        checkpoint,
        distillate_time):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned
        criterion: loss criterion
        optimizer: optimizer to use in training
        optim_name: optimizer name used to decay and drop
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter
        scheduler: set learning rate to drop after x steps
        debug_mode: if true then no log and checkpoint will be created
        pretrained: for logging name only
        freeze: for logging name only
        checkpoint: for logging name only
        distillate_time: distillate_time file
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """

    checkpoint_path = ''

    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    csv_time = datetime.datetime.now().isoformat()
    csv_model = model.name
    csv_optim = optim_name
    csv_lr = LR
    csv_step = scheduler.step_size

    if not os.path.exists('results_csv'):
        os.makedirs('results_csv')

    # List used for csv later
    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_val_acc = []
    list_notes = []

    create_csv(debug_mode, pretrained, freeze, csv_time, csv_model, csv_optim, csv_lr, csv_step,
               list_train_loss, list_train_acc, list_val_loss, list_val_acc,
               list_notes, checkpoint)

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')

        # training and val status
        training_print = 0
        val_print = 0

        # loading bar
        loading_bar = ''
        dataloaders_length = len(dataloaders['train']) + len(dataloaders['val'])
        for i in range(dataloaders_length):
            loading_bar += '-'
 
        distillate_index = 0
        if distillate_time != '':
            distillate_results = np.load(f'results_distillation/d-{distillate_time}.npy')

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            total_done = 0

            # reset output_acc
            output_acc = 0

            num_of_steps = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                inputs, labels = data
                batch_size = inputs.shape[0]

                # loading bar progress
                loading_bar = f'={loading_bar}'
                loading_bar = loading_bar[:dataloaders_length]
                print(f'Steps: {loading_bar}', end='\r')

                # ===========================================================
                # train datasets
                if phase == "train":
                    for i in range(inputs.size()[1]):
                        optimizer.zero_grad()

                        inp = inputs.clone()[:, i]
                        inp = inp.cuda()
                        labels = labels.cuda()
                        num_of_steps += 1
                        outputs = model(inp)

                        if distillate_time != '':
                            labels = get_distillate_output(distillate_results, distillate_index, outputs.shape[0])
                        if(isinstance(outputs, list)):
                            current_loss = []

                            # get output pred and get accuracy
                            for output_item in outputs:
                                if distillate_time != '':
                                    current_loss.append(criterion(outputs, get_distillate_output(distillate_results, distillate_index, outputs.shape[0])))
                                else:
                                    current_loss.append(criterion(output_item, labels))

                            combined_pred = torch.max(outputs[-1], dim=1)[1]
                            output_acc += accuracy_score(labels.cpu().data.numpy(), combined_pred.cpu().data.numpy())

                            loss = sum(current_loss[0:-1]) / len(current_loss[0:-1]) + (current_loss[-1] / 2)
                        # if outputs.size is 3 dimensional from new resatt
                        elif(len(outputs.size()) == 3):
                            current_loss = []

                            for output_item in outputs:
                                current_loss.append(criterion(output_item, labels))

                            loss = sum(current_loss)
                            outputs_pred = torch.max(outputs[0], dim=1)[1]
                            output_acc += accuracy_score(labels.cpu().data.numpy(), outputs_pred.cpu().data.numpy())
                        else:
                            # get output pred and get accuracy
                            if distillate_time != '':
                                loss = criterion(outputs, get_distillate_output(distillate_results, distillate_index, outputs.shape[0]))
                            else:
                                loss = criterion(outputs, labels)

                            outputs_pred = torch.max(outputs, dim=1)[1]
                            output_acc += accuracy_score(labels.cpu().data.numpy(), outputs_pred.cpu().data.numpy())

                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * batch_size
                # ===========================================================
                # val datasets
                # else:
                elif phase == "val":
                    optimizer.zero_grad()
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)

                    if(isinstance(outputs, list)):
                        current_loss = []

                        # get output pred and get accuracy
                        for output_item in outputs:
                            if distillate_time != '':
                                current_loss.append(criterion(outputs, get_distillate_output(distillate_results, distillate_index, outputs.shape[0])))
                            else:
                                current_loss.append(criterion(output_item, labels))

                        combined_pred = torch.max(outputs[-1], dim=1)[1]
                        
                        output_acc += accuracy_score(labels.cpu().data.numpy(), combined_pred.cpu().data.numpy())
                        loss = sum(current_loss[0:-1]) / len(current_loss[0:-1]) + (current_loss[-1] / 2)
                    elif(len(outputs.size()) == 3):
                        for output_item in outputs:
                            current_loss.append(criterion(output_item, labels))

                        loss = sum(current_loss)
                        outputs_pred = torch.max(outputs[0], dim=1)[1]
                        output_acc += accuracy_score(labels.cpu().data.numpy(), outputs_pred.cpu().data.numpy())
                    else:
                        if distillate_time != '':
                            loss = criterion(outputs, get_distillate_output(distillate_results, distillate_index, outputs.shape[0]))
                        else:
                            loss = criterion(outputs, labels)

                        # get output pred and get accuracy
                        outputs_pred = torch.max(outputs, dim=1)[1]

                        output_acc += accuracy_score(labels.cpu().data.numpy(), outputs_pred.cpu().data.numpy())

                    running_loss += loss.item() * batch_size

                distillate_index += 1
                # ===========================================================

            for data in dataloaders['train']:
                dataloaders_crop_count, _ = data
                break
            if phase == 'train':
                output_acc = output_acc / len(dataloaders[phase]) / dataloaders_crop_count.size()[1]
                epoch_loss = running_loss / dataset_sizes[phase] / dataloaders_crop_count.size()[1]
                last_train_loss = epoch_loss
                training_print = f'{phase} epoch {epoch}: acc {output_acc:.2f} loss {epoch_loss:.4f} with data size {dataset_sizes[phase]}'

                list_train_loss.append(f'{epoch_loss:.4f}')
                list_train_acc.append(f'{output_acc:.2f}')
            else:
                output_acc = output_acc / len(dataloaders[phase])
                epoch_loss = running_loss / dataset_sizes[phase]
                val_print = f'{phase} epoch {epoch}: acc {output_acc:.2f} loss {epoch_loss:.4f} with data size {dataset_sizes[phase]}'

                list_val_loss.append(f'{epoch_loss:.4f}')
                list_val_acc.append(f'{output_acc:.2f}')
                print('')

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss and not debug_mode:
                best_loss = epoch_loss
                best_epoch = epoch

                checkpoint_path = create_checkpoint(model, best_loss, epoch, pretrained, freeze, since, csv_model, csv_optim, csv_lr, csv_step, list_train_loss, list_train_acc, list_val_loss, list_val_acc)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        # decay learning rate every x steps
        scheduler.step()
        if epoch % scheduler.step_size == 0:
            lr_new = '{0:.20f}'.format(optimizer.param_groups[0]['lr']).rstrip('0')
            optim_print = f'created new {optim_name} optimizer with LR: {lr_new}'
            print(optim_print)
            list_notes.append(optim_print)
        else:
            list_notes.append('')

        # update csv report
        create_csv(debug_mode, pretrained, freeze, csv_time, csv_model, csv_optim, csv_lr, csv_step,
                   list_train_loss, list_train_acc, list_val_loss, list_val_acc,
                   list_notes, checkpoint)

        print(training_print)
        print(val_print)

        data = {'train_loss': list_train_loss, 'train_acc': list_train_acc}

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        # if ((epoch - best_epoch) >= 3):
        #     print("no improvement in 3 epochs, break")
        #     break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    # checkpoint_best = torch.load(checkpoint_path)
    # model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(MODEL_NAME, PRETRAINED, FREEZE, EPOCHS, BATCH_SIZE, N_LABELS, OPTIMIZERS, PATH_TO_IMAGES, LR, WEIGHT_DECAY, LR_DECAY_STEPS, DEBUG_MODE, CHECKPOINT_PATH = '', DISTILLATE_WITH = '', CUSTOM_LABEL = ''):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        MODEL_NAME: model name
        PRETRAINED: if model pretrained
        FREEZE: model layer frozen or not
        EPOCHS: epochs iteration
        BATCH_SIZE: number of batch data per training
        N_LABELS: number of class labels
        OPTIMIZERS: optimizers used
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD
        LR_DECAY_STEPS: how many steps before LR decayed and dropped
        DEBUG_MODE: if true then no log will be created
        CHECKPOINT_PATH: load checkpoint path
        DISTILLATE_WITH: distillate the model with
    Returns:
        # preds: torchvision model predictions on test fold with ground truth for comparison
        # aucs: AUCs for each train,test tuple

    """

    if not os.path.exists('results'):
        os.makedirs('results')

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    # define torchvision transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # create train/val dataloaders
    transformed_datasets = {x: datasets.ImageFolder(os.path.join(PATH_TO_IMAGES, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(transformed_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # Check model used
    import modified_densenet
    import modified_alexnet

    model = (models.densenet121(pretrained=PRETRAINED) if MODEL_NAME == 'densenet' 
            else modified_densenet.densenet121(type=MODEL_NAME, pretrained=PRETRAINED) 
                if MODEL_NAME == 'va-densenet' 
                or MODEL_NAME == 'reva-densenet' 
                or MODEL_NAME == 'fp-densenet' 
                or MODEL_NAME == 'start-densenet' 
                or MODEL_NAME == 'every-densenet' 
                or MODEL_NAME == 'sedensenet'
                or MODEL_NAME == 'triplelossdensenet'
            else models.alexnet(pretrained=PRETRAINED) if MODEL_NAME == 'alexnet'
            else modified_alexnet.alexnet(type=MODEL_NAME, pretrained=PRETRAINED) 
                if MODEL_NAME == 'va-alexnet'
                or MODEL_NAME == 'reva-alexnet'
                or MODEL_NAME == 'fp-alexnet'
                or MODEL_NAME == 'start-alexnet'
                or MODEL_NAME == 'every-alexnet'
            else models.resnet152(pretrained=PRETRAINED)if MODEL_NAME == 'resnet'
            else models.vgg16(pretrained=PRETRAINED)if MODEL_NAME == 'VGG'
            else models.vgg16_bn(pretrained=PRETRAINED)if MODEL_NAME == 'VGG_Bn'
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
                    or MODEL_NAME == 'every-alexnet' 
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
    elif (MODEL_NAME == 'alexnet' or MODEL_NAME == 'va-alexnet' or MODEL_NAME == 'va-alexnet' or MODEL_NAME == 'reva-alexnet' or MODEL_NAME == 'fp-alexnet' or MODEL_NAME == 'start-alexnet' or MODEL_NAME == 'every-alexnet' or MODEL_NAME == 'VGG' or MODEL_NAME == 'VGG_Bn'):
        model.classifier[6] = nn.Linear(num_ftrs, N_LABELS)
    elif (MODEL_NAME == 'resnet'):
        model.fc = nn.Linear(num_ftrs, N_LABELS)
    else:
        raise ValueError("Error model name")

    if MODEL_NAME == 'every-densenet':
        model.classifier1 = nn.Linear(model.classifier1.in_features, N_LABELS)
        model.classifier2 = nn.Linear(model.classifier2.in_features, N_LABELS)
        model.classifier3 = nn.Linear(model.classifier3.in_features, N_LABELS)

    if CHECKPOINT_PATH != '':
        model = load_checkpoint(model, CHECKPOINT_PATH)

    # show params to learn
    if FREEZE:
        for name, param in model.named_parameters():
            attention_pattern = re.compile(r'^(conv2d1x1|valinear|transconv|start|every|se_).+$')
            classifier_pattern = re.compile(r'^(classifier(?!\.\d)|classifier\.6|fc).+$')
            if attention_pattern.match(name):
                param.requires_grad = True
            elif classifier_pattern.match(name) and CHECKPOINT_PATH == '':
                param.requires_grad = True
            else:
                param.requires_grad = False

    if FREEZE:
        print('Params to learn:')
        for name, param in model.named_parameters():
            if param.requires_grad == True:
               print(name)
        print('==================================')


    # Distillate
    distillate_time = ''
    if DISTILLATE_WITH != '':
        print(f'Distillate with {DISTILLATE_WITH}')
        distillate_time = datetime.datetime.now().isoformat()
        model_distillate = models.densenet121(pretrained=PRETRAINED)
        num_ftrs_distillate = model_distillate.classifier.in_features
        model_distillate.classifier = nn.Linear(num_ftrs_distillate, N_LABELS)
        model_distillate = load_checkpoint(model_distillate, DISTILLATE_WITH)
        print('Loaded checkpoint for distillation')
        model_distillate = model_distillate.cuda()

        loading_bar = ''
        dataloaders_length = len(dataloaders['train']) + len(dataloaders['val'])
        for i in range(dataloaders_length):
            loading_bar += '-'

        for phase in ['train', 'val']:
            for data in dataloaders[phase]:
                loading_bar = f'={loading_bar}'
                loading_bar = loading_bar[:dataloaders_length]
                print(f'Distillating: {loading_bar}', end='\r')

                inputs, labels = data
                if phase == 'train':
                    for i in range(10):
                        inp = inputs.clone()[:, i]
                        inp = inp.cuda()
                        labels = labels.cuda()
                        outputs = model_distillate(inp).cpu().data.numpy()
                        if len(outputs) != BATCH_SIZE:
                            outputs_padding = np.zeros((BATCH_SIZE, N_LABELS))
                            outputs_padding[:outputs.shape[0], :outputs.shape[1]] = outputs
                            outputs = outputs_padding
                        if Path(f'results_distillation/d-{distillate_time}.npy').exists():
                            loaded_np = np.load(f'results_distillation/d-{distillate_time}.npy')
                            outputs = np.append(loaded_np, [ outputs ], axis=0)
                        else:
                            outputs = [ outputs ]
                        np.save(f'results_distillation/d-{distillate_time}.npy', outputs)
                else:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = model_distillate(inputs).cpu().data.numpy()
                    if len(outputs) != BATCH_SIZE:
                        outputs_padding = np.zeros((BATCH_SIZE, N_LABELS))
                        outputs_padding[:outputs.shape[0], :outputs.shape[1]] = outputs
                        outputs = outputs_padding
                    loaded_np = np.load(f'results_distillation/d-{distillate_time}.npy')
                    outputs = np.append(loaded_np, [ outputs ], axis=0)
                    np.save(f'results_distillation/d-{distillate_time}.npy', outputs)
        print('')

    # put model on GPU
    model = model.cuda()
    model.name = MODEL_NAME

    # define criterion, optimizer for training
    if distillate_time != '':
        criterion = nn.MSELoss()
    else:
        # class_weight_value = [
            # 1 - (1 / 77),
            # 1 - (3 / 77),
            # 1 - (12 / 77),
            # 1 - (19 / 77),
            # 1 - (3 / 77),
            # 1 - (33 / 77),
            # 1 - (6 / 77)
        # ]
        # class_weight = torch.FloatTensor(class_weight_value).cuda()
        # criterion = nn.CrossEntropyLoss(weight=class_weight)
        criterion = nn.CrossEntropyLoss()

    # Check if SGD or Adam
    optimizer = (optim.SGD(model.parameters(), lr=LR, momentum=0.9) if OPTIMIZERS == 'SGD'
                 else optim.Adam(model.parameters(), lr=LR) if OPTIMIZERS == 'Adam'
                 else '')

    scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEPS, gamma=0.1)

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}


    model.name = model.name + '-' + CUSTOM_LABEL
    print(model.name)

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, optim_name=OPTIMIZERS,
                                    LR=LR, num_epochs=EPOCHS, dataloaders=dataloaders,
                                    dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,
                                    scheduler=scheduler, debug_mode=DEBUG_MODE, pretrained=PRETRAINED, freeze=FREEZE,
                                    checkpoint=CHECKPOINT_PATH, distillate_time=distillate_time)

    print("Finished Training")

    # # get preds and AUCs on test fold
    # preds, aucs = E.make_pred_multilabel(
    #     data_transforms, model, PATH_TO_IMAGES)

    # return preds, aucs
