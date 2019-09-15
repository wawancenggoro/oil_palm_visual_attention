import os
import re
import sys
import numpy as np
import torch
import torchvision.utils as vutils
import shutil
from PIL import Image

title_arg = sys.argv[1]
folder_pattern = re.compile(f'^{title_arg}.+')

for foldername in sorted(os.listdir('attention_image')):
    if folder_pattern.match(foldername):
        numpy_name = foldername.split('_')[2]

        title_pattern = re.compile(f'^{numpy_name}.+\.npy')
        org_pattern = re.compile(f'^org_{numpy_name}\.npy')

        if os.path.exists(f'attention_image/result/{foldername}'):
            shutil.rmtree(f'attention_image/result/{foldername}')
        os.makedirs(f'attention_image/result/{foldername}')

        for filename in os.listdir(f'attention_image/{foldername}'):
            if title_pattern.match(filename):
                numpy_attention = np.load(f'attention_image/{foldername}/{filename}')
                attention = torch.from_numpy(numpy_attention).float().to('cuda:0')
                # attention_global = torch.from_numpy(numpy_attention).float().to('cuda:0').sum(0)

                # max_global = attention_global.max().cpu().data.numpy()
                # image_global = attention_global / (max_global * 1 if max_global != 0 else 1) * 255
                # vutils.save_image(image_global, f'attention_image/result/{title_arg}_global.jpg', normalize=True)

                for i in range(attention.size()[0]):
                    max_local = attention[i].max().cpu().data.numpy()
                    image_local = attention[i] / (max_local * 1 if max_local != 0 else 1) * 255
                    image_name = f"{filename.split('_')[1].split('.')[0]}_{i}.jpg"
                    vutils.save_image(image_local, f'attention_image/result/{foldername}/{image_name}', normalize=True)
            elif org_pattern.match(filename):
                numpy_image = np.load(f'attention_image/{foldername}/{filename}')
                org_image = torch.from_numpy(numpy_image).float().to('cuda:0')
                image_name = f"0_org_{filename.split('_')[1].split('.')[0]}.jpg"
                vutils.save_image(org_image, f'attention_image/result/{foldername}/{image_name}', normalize=True)
