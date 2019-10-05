import os
import re
import cv2

FOLDER_NAME = 'attention_image/result'

FILE_COUNT = 0
for foldername in sorted(os.listdir(FOLDER_NAME)):
    originalimage_pattern = re.compile('^0_org.+')
    reg_pattern = re.compile('.+jet\\.jpg$')
    filename_pattern = re.compile(f'.+[^\\.jpg]')
    
    original_filename = ''
    for filename in sorted(os.listdir(f'{FOLDER_NAME}/{foldername}/')):
        if originalimage_pattern.match(filename):
            original_filename = originalimage_pattern.match(filename)[0]

    if original_filename is not '':
        for filename in sorted(os.listdir(f'{FOLDER_NAME}/{foldername}/')):
            extracted_filename = filename_pattern.match(filename)[0]
            if not reg_pattern.match(filename) and not os.path.exists(f'{FOLDER_NAME}/{foldername}/{extracted_filename}-jet.jpg'):
                img = cv2.imread(f'{FOLDER_NAME}/{foldername}/{filename}', 1)
                bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                heatmap_img = cv2.applyColorMap(bw_img, cv2.COLORMAP_JET)
                original_image = cv2.imread(f'{FOLDER_NAME}/{foldername}/{original_filename}', 1)

                combined_img = cv2.addWeighted(heatmap_img, 0.7, original_image, 0.3, 0)
                cv2.imwrite(f'{FOLDER_NAME}/{foldername}/{extracted_filename}-jet.jpg', combined_img)
    print(f'{foldername} done heatmapped')




# for foldername in sorted(os.listdir('attention_image/')):
    # reg_pattern = re.compile('.+jet\\.jpg$')
    # for filename in sorted(os.listdir(f'attention_image/{foldername}/')):
        # if reg_pattern.match(filename):
            # os.remove(f'attention_image/{foldername}/{filename}')
