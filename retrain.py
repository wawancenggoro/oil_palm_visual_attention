import model as M
import test_model as N
from threading import Thread

DEBUG_MODE = True

alexnet = "alexnet"
va_alexnet = "va-alexnet" 
reva_alexnet = "reva-alexnet"
fp_alexnet = "fp-alexnet"
start_alexnet = "start-alexnet"
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
PATH_TO_IMAGES = "./NEW_DATASET_AUGMENTED"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY_STEPS = 16

EPOCHS = 50
BATCH_SIZE = 16
N_LABELS = 7
# FREEZE = True
PRETRAINED = True

# CHECKPOINT_TIMESTAMP = '1560599355'
CHECKPOINT_TIMESTAMP = ''

# DISTILLATE_WITH = '1560599355'
DISTILLATE_WITH = ''

# alexnet, va_alexnet, reva_alexnet, fp_alexnet, startva_alexnet,
# densenet, va_densenet, reva_densenet, fp_densenet, start_densenet, every_densenet, sedensenet, triplelossdensenet
# resnet, VGG, VGG_Bn
for _, item in enumerate([ 
                          # [densenet, '1564583958'],
                          # [alexnet, '1564862020'],
                          # [sedensenet, '1564680042'],
                          # [every_densenet, '1570275876'],

                          [every_densenet, '1570434014'],
                          [every_densenet, '1570509863'],

                          # [densenet, '1570267775'],
                          # [densenet, '1570425915'],
                          # [densenet, '1570444049'],
                          # [densenet, '1570462212'],
                          # [densenet, '1570462236'],
                          # [densenet, '1570495656'],
                          # [densenet, '1570495684'],

                          # [every_densenet, '1565114071'],
                          # [every_densenet, '1565142577'],
                          # [fp_densenet, '1564683135'],
                        ]):
    for i in range(1):
        i = 1
        # N.test_cnn(item[0], densenet, BATCH_SIZE, N_LABELS, PATH_TO_IMAGES, DEBUG_MODE, item[1], '1570444049')

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
            print(f"CHECKPOINT_TIMESTAMP = {CHECKPOINT_TIMESTAMP}")
            # # print(f"CHECKPOINT_TEST = {CHECKPOINT_TEST}")
            # # print(f"CHECKPOINT_TEST_TARGET = {CHECKPOINT_TEST_TARGET}")
            # # print(f"CHECKPOINT_TEST = {CHECKPOINT_TEST}")
            # # print(f"CHECKPOINT_TEST_TARGET = {CHECKPOINT_TEST_TARGET}")
            print("============================================")

            M.train_cnn(MODEL_NAME, PRETRAINED, FREEZE, EPOCHS, BATCH_SIZE, N_LABELS, OPTIMIZERS, PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, LEARNING_RATE_DECAY_STEPS, DEBUG_MODE, CHECKPOINT_TIMESTAMP, DISTILLATE_WITH)
