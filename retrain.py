import os
from os import walk
from shutil import copy2

import model as M
import test_model as N
from threading import Thread

DEBUG_MODE = True

alexnet = "alexnet"
va_alexnet = "va-alexnet" 
reva_alexnet = "reva-alexnet"
fp_alexnet = "fp-alexnet"
start_alexnet = "start-alexnet"
every_alexnet = "every-alexnet"
densenet = "densenet"
va_densenet = "va-densenet"
reva_densenet = "reva-densenet"
fp_densenet = "fp-densenet"
start_densenet = "start-densenet"
every_densenet = "every-densenet"
sedensenet = "sedensenet"
triplelossdensenet = "triplelossdensenet"
resnet = "resnet"
VGG = "VGG"
VGG_Bn = "VGG-Bn"

Adam = "Adam"
SGD = "SGD"

# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "./NEW_DATASET"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_STEPS = 8

EPOCHS = 50
BATCH_SIZE = 8
N_LABELS = 7
# FREEZE = True
PRETRAINED = True

# CHECKPOINT_TIMESTAMP = '1560599355'
# CHECKPOINT_TIMESTAMP = '1564583958'
CHECKPOINT_TIMESTAMP = ''
CHECKPOINT_TEST = ''
# CHECKPOINT_TEST = '1563112018'
# CHECKPOINT_TEST_TARGET = '1563032942'

# DISTILLATE_WITH = '1560599355'
DISTILLATE_WITH = ''

CUSTOM_LABEL = 'CV-set-'
# ==================================================================

# alexnet, va_alexnet, reva_alexnet, fp_alexnet, startva_alexnet,
# densenet, va_densenet, reva_densenet, fp_densenet, start_densenet, every_densenet, sedensenet, triplelossdensenet
# resnet, VGG, VGG_Bn
for MODEL_NAME in [every_densenet]:
    for OPTIMIZERS in [SGD]:
        for FREEZE in [False]:
            print("============================================")
            print(f"LEARNING_RATE = {LEARNING_RATE}")
            print(f"LEARNING_RATE_DECAY_STEPS = {LEARNING_RATE_DECAY_STEPS}")
            print(f"MODEL_NAME = {MODEL_NAME}")
            print(f"PRETRAINED = {PRETRAINED}")
            print(f"FREEZE = {FREEZE}")
            print(f"EPOCHS = {EPOCHS}")
            print(f"BATCH_SIZE = {BATCH_SIZE}")
            print(f"N_LABELS = {N_LABELS}")
            print(f"OPTIMIZERS = {OPTIMIZERS}")
            if (CHECKPOINT_TIMESTAMP != ''):
                print(f"CHECKPOINT_TIMESTAMP = {CHECKPOINT_TIMESTAMP}")
            if (CHECKPOINT_TEST != ''):
                print(f"CHECKPOINT_TEST = {CHECKPOINT_TEST}")
                print(f"CHECKPOINT_TEST_TARGET = {CHECKPOINT_TEST_TARGET}")
            print(f"CUSTOM_LABEL = {CUSTOM_LABEL}")
            print("============================================")

            # N.test_cnn(every_densenet, densenet, BATCH_SIZE, N_LABELS, PATH_TO_IMAGES, DEBUG_MODE, CHECKPOINT_TEST, CHECKPOINT_TEST_TARGET)
            M.train_cnn(MODEL_NAME, PRETRAINED, FREEZE, EPOCHS, BATCH_SIZE, N_LABELS, OPTIMIZERS, PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, LEARNING_RATE_DECAY_STEPS, DEBUG_MODE, CHECKPOINT_TIMESTAMP, DISTILLATE_WITH, CUSTOM_LABEL)

            # for i in range(5):
                # M.train_cnn(MODEL_NAME, PRETRAINED, FREEZE, EPOCHS, BATCH_SIZE, N_LABELS, OPTIMIZERS, f'{PATH_TO_IMAGES}/set_{i}', LEARNING_RATE, WEIGHT_DECAY, LEARNING_RATE_DECAY_STEPS, DEBUG_MODE, CHECKPOINT_TIMESTAMP, DISTILLATE_WITH, f'{CUSTOM_LABEL}{i}')
