#script to test and implement from urllib.request import urlopen
#https://huggingface.co/timm/maxvit_tiny_tf_512.in1k
from urllib.request import urlopen
from PIL import Image
import timm
import torch
#import torchvision


#net = torchvision.models.resnet18(pretrained=True)
#for name, param in net.named_parameters():
#    print (name, param.requires_grad)

#for (name, module) in net.named_children():
#     print(name)


import os
import csv
train_dir = '/mnt/recsys/daniel/simase_network/childadult_train/train'

with open('/mnt/recsys/daniel/simase_network/childadult_train/csv_files/train.csv', 'a') as file:

    writerObj = csv.writer(file)
    writerObj.writerow(['Filen_name','label','class'])
    for root, dirs, files in os.walk(train_dir):
        for filename in files:
            if filename.endswith('.png'):
                label = root.split('/')[-1]
                print (label)
                if label == 'child':
                    int_value = 0
                elif label == 'adult':
                    int_value = 1
                writerObj.writerow([filename, str(int_value), label])


                #print(os.path.join(root, filename))
            

