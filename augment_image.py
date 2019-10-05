import os
import re
import cv2

FOLDER_NAME = 'NEW_DATASET_AUGMENTED'

FILE_COUNT = 0

# FILENAME_FULL_PATH = []
# FILENAME_FULL_PATH.append(f'{FOLDER_NAME}/{foldername}/{classname}/{filename}')
for foldername in sorted(os.listdir(FOLDER_NAME)):
    for classname in sorted(os.listdir(f'{FOLDER_NAME}/{foldername}')):
        for filename in sorted(os.listdir(f'{FOLDER_NAME}/{foldername}/{classname}')):
            filename_pattern = re.compile(f'.+[^\\.jpg]')
    
            original_filename = filename_pattern.match(filename)[0]

            img = cv2.imread(f'{FOLDER_NAME}/{foldername}/{classname}/{filename}', 1)
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_LINEAR)
            img_h = cv2.flip(img, 0)
            img_v = cv2.flip(img, 1)

            img_r90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img_r180 = cv2.rotate(img_r90, cv2.ROTATE_90_CLOCKWISE)
            img_r270 = cv2.rotate(img_r180, cv2.ROTATE_90_CLOCKWISE)
            img_h_r90 = cv2.rotate(img_h, cv2.ROTATE_90_CLOCKWISE)
            img_h_r180 = cv2.rotate(img_h_r90, cv2.ROTATE_90_CLOCKWISE)
            img_h_r270 = cv2.rotate(img_h_r180, cv2.ROTATE_90_CLOCKWISE)
            img_v_r90 = cv2.rotate(img_v, cv2.ROTATE_90_CLOCKWISE)
            img_v_r180 = cv2.rotate(img_v_r90, cv2.ROTATE_90_CLOCKWISE)
            img_v_r270 = cv2.rotate(img_v_r180, cv2.ROTATE_90_CLOCKWISE)

            write_target_folder = f'{FOLDER_NAME}/{foldername}/{classname}'
            cv2.imwrite(f'{write_target_folder}/{original_filename}.jpg', img)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_r90.jpg', img_r90)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_r180.jpg', img_r180)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_r270.jpg', img_r270)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_h.jpg', img_h)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_h_r90.jpg', img_h_r90)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_h_r180.jpg', img_h_r180)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_h_r270.jpg', img_h_r270)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_v.jpg', img_v)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_v_r90.jpg', img_v_r90)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_v_r180.jpg', img_v_r180)
            cv2.imwrite(f'{write_target_folder}/{original_filename}_v_r270.jpg', img_v_r270)
            print(f'Done {original_filename}')
        print(f'Done {classname}')
    print(f'Done {foldername}')

