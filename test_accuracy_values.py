import _init_paths
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import cv2
from model.net import get_model
from dataloader.triplet_img_loader import get_loader
from utils.gen_utils import make_dir_if_not_exist
from utils.vis_utils import vis_with_paths, vis_with_paths_and_bboxes
import wandb
import numpy as np
from config.base_config import cfg, cfg_from_file
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from PIL import Image
import csv
class DatasetClass(Dataset):
    def __init__(self, annotations_file,  img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,  self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 0])

        image = Image.open(img_path)
        #print (img_path)
        #image = cv2.resize(image, (228, 228))
        label = self.img_labels.iloc[idx, 1]
        img_name = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

def test(data, model, device):
    phase = 'test'

    running_corrects = 0
    accuracy_csv_file = '/mnt/recsys/daniel/simase_network/CNN_baseline/cleft_lip_embedding.csv'
    with open(accuracy_csv_file, mode='w') as file:
        file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(["Img_name", "Class0", 'Class1', 'Class2', 'label'])

    for inputs, labels, img_name  in data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.to(memory_format=torch.contiguous_format)
        # zero the parameter gradients
        # track history if only in trainR

        outputs = model(inputs)
        print_embedding = outputs.clone().cpu().detach().numpy()
        label_print = labels.clone().cpu().detach().numpy()
        with open(accuracy_csv_file, mode='a') as file:
            file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow([img_name, print_embedding[0][0], print_embedding[0][1],  print_embedding[0][2], label_print[0]])

        _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
        # statistics

        running_corrects += torch.sum(preds == labels.data)

        # log_step = args.train_log_step
        # if (batch_idx % log_step == 0) and (batch_idx != 0):
        #    print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
        #    total_loss = 0

    test_epoch_acc = (running_corrects.double() / len(data.dataset)).cpu().detach().numpy()


    return test_epoch_acc

cuda = True
cuda_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)
# cudnn.benchmark = True #optimizes when the input size of the network is same. dont use when the input sizes are different.
if cuda:
    device = 'cuda'


else:
    device = 'cpu'
test_data_dir = '/mnt/recsys/daniel/simase_network/cleft_lip_data_600_800/test'
test_annot_file = '/mnt/recsys/daniel/simase_network/cleft_lip_data_600_800/csv_files/test_file.csv'
pretrained_model_path = '/mnt/recsys/daniel/simase_network/CNN_baseline/experiments/cleftLip/vgg16/baseline/checkpoint_10.pth'

# dataloader classes and the transforms are defined here. More transforms should be tried


transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = DatasetClass(test_annot_file, test_data_dir, transform)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = models.vgg16()
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
model = model.to(device)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
'''
if os.path.isfile(pretrained_model_path):
    print("=> Loading checkpoint '{}'".format(pretrained_model_path))
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> Loaded checkpoint '{}'".format(pretrained_model_path))
else:
    print("=> No checkpoint found at '{}'".format(pretrained_model_path))
'''

model.eval()  # Set model to evaluate mode
running_loss = 0.0
running_corrects = 0


test_epoch_acc = test(test_data_loader, model, device)

#test_loss, accuracy_zero_margin = test(test_data_loader, model, criterion)
print (test_epoch_acc)



