import os

CURRENTDIR = os.getcwd()

DATADIR = os.path.join(CURRENTDIR, "NEW_DATASET/val")
TARGETDIR = os.path.join(CURRENTDIR, "NEW_DATASET/test")
CATEGORIES = []

CATEGORIES = [x for x in sorted(os.listdir(DATADIR))]
print(CATEGORIES)

if not os.path.exists(TARGETDIR):
    os.mkdir(TARGETDIR)

for CATEGORY in CATEGORIES:
    PATH = os.path.join(DATADIR, CATEGORY)
    TARGETPATH = os.path.join(TARGETDIR, CATEGORY)

    if not os.path.exists(TARGETPATH):
        os.mkdir(TARGETPATH)

    NUM_OF_FILES = int(len(os.listdir(PATH)) / 2)
    i = 1
    for FILE in os.listdir(PATH):
        os.rename(os.path.join(PATH, FILE), os.path.join(TARGETPATH, FILE))
        if i < NUM_OF_FILES:
            i += 1
        else:
            break
