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

class DatasetClass(Dataset):
    def __init__(self, annotations_file,  img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,  self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 0])

        image = cv2.imread(img_path)
        #print (img_path)
        image = cv2.resize(image, (228, 228))
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return image, label

def main():

    cuda = True
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)
    # cudnn.benchmark = True #optimizes when the input size of the network is same. dont use when the input sizes are different.

    test_data_dir =
    test_annot_file =
    pretrained_model_path =
    device =
    # dataloader classes and the transforms are defined here. More transforms should be tried


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = DatasetClass(test_annot_file, test_data_dir, transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)



    # Build Model
    model = get_model( device)

    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0


    test_epoch_loss, test_epoch_acc = test(test_data_loader, model)

    #test_loss, accuracy_zero_margin = test(test_data_loader, model, criterion)
    print ()



def test(data, model, device):
    phase = 'test'
    print("******** Training ********")
    running_corrects = 0

    for inputs, labels in data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.to(memory_format=torch.contiguous_format)
        # zero the parameter gradients
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
        # statistics

        running_corrects += torch.sum(preds == labels.data)

        # log_step = args.train_log_step
        # if (batch_idx % log_step == 0) and (batch_idx != 0):
        #    print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
        #    total_loss = 0

    test_epoch_acc = running_corrects.double() / len(data.dataset)
    print(  'Acc  :{}').format(test_epoch_acc)

    return test_epoch_acc