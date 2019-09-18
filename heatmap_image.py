import os
import re
import cv2

for foldername in sorted(os.listdir('attention_image/')):
    originalimage_pattern = re.compile('^0_org.+')
    reg_pattern = re.compile('.+jet\\.jpg$')
    filename_pattern = re.compile(f'.+[^\\.jpg]')
    
    original_filename = ''
    for filename in sorted(os.listdir(f'attention_image/{foldername}/')):
        if originalimage_pattern.match(filename):
            original_filename = originalimage_pattern.match(filename)[0]

    if original_filename is not '':
        for filename in sorted(os.listdir(f'attention_image/{foldername}/')):
            extracted_filename = filename_pattern.match(filename)[0]

            img = cv2.imread(f'attention_image/{foldername}/{filename}', 1)
            bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            heatmap_img = cv2.applyColorMap(bw_img, cv2.COLORMAP_JET)
            original_image = cv2.imread(f'attention_image/{foldername}/{original_filename}', 1)

            combined_img = cv2.addWeighted(heatmap_img, 0.7, original_image, 0.3, 0)
            cv2.imwrite(f'attention_image/{foldername}/{extracted_filename}-jet.jpg', combined_img)




# for foldername in sorted(os.listdir('attention_image/')):
    # reg_pattern = re.compile('.+jet\\.jpg$')
    # for filename in sorted(os.listdir(f'attention_image/{foldername}/')):
        # if reg_pattern.match(filename):
            # os.remove(f'attention_image/{foldername}/{filename}')
