import os
import cv2

folder_path = '/home/daniel/simase_network/Cleft_lip_data/test'

new_dir = '/home/daniel/simase_network/cleft_lip_data_600_800/test'


sub_folders = os.listdir(folder_path)
big_list = []
for k in sub_folders:
    folder = os.path.join(folder_path,k)
    small_list = []
    imgs = os.listdir(folder)
    for img_name in imgs:
        if img_name.endswith("png"):
            img = cv2.imread(os.path.join(folder,img_name))
            h, w, _ = img.shape
            resized_image = cv2.resize(img, (800 ,600), interpolation = cv2.INTER_AREA)
            ori_path = os.path.join(folder_path, k, img_name)
            save_path = os.path.join(new_dir, k, img_name)
            cv2.imwrite(save_path, resized_image)

print('DOne!')





