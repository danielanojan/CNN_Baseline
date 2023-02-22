from torchvision.transforms import InterpolationMode

import _init_paths
import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import cv2
from model.net import get_model, NeuralNet
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
import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from torchvision import datasets, models, transforms
import csv

class DatasetClass(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,  self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 0])

        image = Image.open(img_path)
        # print (img_path)
        # image = cv2.resize(image, (228, 228))
        label = self.img_labels.iloc[idx, 1]
        img_name = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label, img_name



def main():

    cuda_id = 0

    if args.cuda:
        device = 'cuda'
        torch.cuda.set_device(cuda_id)
    else:
        device = 'cpu'
    # cudnn.benchmark = True #optimizes when the input size of the network is same. dont use when the input sizes are different.
    rootdir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(rootdir, os.pardir))
    exp_dir = os.path.join(rootdir, 'experiments', args.exp, args.archi, args.exp_type )
    ckp_path = os.path.join(exp_dir, args.ckp_file)

    train_annot_file = os.path.join(parent_dir, args.dataset, 'csv_files/train_file.csv')
    test_annot_file = os.path.join(parent_dir, args.dataset, 'csv_files/test_file.csv')
    train_data_dir = os.path.join(parent_dir, args.dataset, 'train')
    test_data_dir = os.path.join(parent_dir, args.dataset, 'test')

    
    if args.archi == 'MaxxViT_tiny_512':
        transform = transforms.Compose([
            transforms.Resize(size=(512, 512), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.CenterCrop(size=(512, 512)),
            transforms.ToTensor()
        ])
    else:

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    # Build Model
    
    model = get_model(args, device, ckp_path)
    if model is None:
        return
    # model = NeuralNet()

    model = model.to(device)



    dataset_emb_train = DatasetClass(train_annot_file, train_data_dir, train_transform)
    data_loader_emb_train = torch.utils.data.DataLoader(dataset_emb_train, batch_size=1, shuffle=False)

    dataset_emb_test = DatasetClass(test_annot_file, test_data_dir, test_transform)
    data_loader_emb_test = torch.utils.data.DataLoader(dataset_emb_test, batch_size=1, shuffle=False)



    embeddings_train, labels_train = generate_emb(data_loader_emb_train, model, device)


    embeddings_test, labels_test = generate_emb(data_loader_emb_test, model, device)



    if args.pca == True:
        epoch = 0
        pca_plot(embeddings_train, labels_train, epoch, exp_dir, label=0000)
        pca_plot(embeddings_test, labels_test, epoch, exp_dir, label=0000)
    if args.confusion_matrix:
        predictions_train = np.argmax(embeddings_train, axis=1)
        conf_matrix_train = confusion_matrix(labels_train, predictions_train, labels=[0,1,2])
        disp1 = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_train)
        disp1.plot()
        #disp.xaxis.set_ticklabels(['healthy','mild','severe'])
        plt.show()

        predictions_test = np.argmax(embeddings_test, axis=1)
        conf_matrix_test = confusion_matrix(labels_test, predictions_test, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test)
        disp.plot()
        # disp.xaxis.set_ticklabels(['healthy','mild','severe'])
        plt.show()

    if args.accuracy_values:
        running_corrects_train = 0
        for emb_train, label_train in zip(embeddings_train, labels_train.reshape(-1, 1)):
            pred = emb_train.argmax()
            if pred==label_train[0]:

                running_corrects_train += 1
        accuracy_train = running_corrects_train / len(data_loader_emb_train)
        print ('Training Accuracy')
        print (accuracy_train)


        running_corrects_test = 0
        for emb_test, label_test in zip(embeddings_test, labels_test.reshape(-1, 1)):
            pred = emb_test.argmax()
            if pred==label_test[0]:

                running_corrects_test += 1
        accuracy_test = running_corrects_test / len(data_loader_emb_test)
        print ('testing Accuracy')
        print (accuracy_test)




def generate_emb(data_loader, model, device):

    model.eval()
    labels = None
    embeddings = None
    inc = 0
    for batch_idx, data in tqdm(enumerate(data_loader)):
        #print (inc)
        inc +=1
        batch_imgs, batch_labels, _ = data
        batch_labels = batch_labels.numpy()
        batch_imgs = Variable(batch_imgs.to(device))
        bacth_E = model(batch_imgs)
        bacth_E = bacth_E.data.cpu().numpy()
        embeddings = np.concatenate((embeddings, bacth_E), axis=0) if embeddings is not None else bacth_E
        labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels


def pca_plot(embeddings, labels, epoch,exp_dir,  label):
    final_data = {
        'embeddings': embeddings,
        'labels': labels
    }
    matrix = "pca"

    if matrix == "pca":
        label_list = pd.DataFrame(labels, columns=["label"])
        pca = PCA(n_components=2)
        principalcomponents = pca.fit_transform(embeddings)
        pcadf = pd.DataFrame(principalcomponents, columns=['pc1', 'pc2'])
        pcadf = pd.concat([pcadf, label_list], axis=1)
        plt.figure(figsize=(16, 9))
        groups = pcadf.groupby('label')
        for name, group in groups:
            plt.plot(group.pc1, group.pc2, marker='o', linestyle='', markersize=10, label=name)
        plt.legend()
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")

        plt.savefig(os.path.join(exp_dir, 'plot_PCA{}_{}.png'.format(label, epoch)))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')

    parser.add_argument('--dataset', default='cleft_lip_data_600_800', type=str,
                        help='[childadult_train/cleft_lip_data_600_800]')
    parser.add_argument('--exp', default='cleftLip', type=str,
                        help='Directory to store results[childAdult/ cleftLip/ cleftLipFT]')
    parser.add_argument('--exp_type', default='adam_baseline', type=str,
                        help='what hyperparameters to choose etc. ')

    parser.add_argument('--archi', type=str, default='vgg16', metavar='M',
                        help='CNN Architectunre[vgg16, alexnet, resnet18, MaxxViT_tiny_512, mobilenet_v3_large, efficientnet_v2_large]')
    parser.add_argument('--num_class', type=int, default=3, metavar='M',
                        help='num of classes in dataset')

    parser.add_argument('--ckp', action='store_true', default=True,
                        help='display option')
    parser.add_argument('--ckp_file', default='checkpoint_12.pth', type=str,
                        help='path to load checkpoint')
    parser.add_argument('--finetune_opt', action='store_true', default=False,
                        help='freezing last layers with opt')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='freezing last layers')

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument("--cuda_id", type=int, default=0,
                        help="List of GPU Devices to train on")

    parser.add_argument('--pca_plot', action='store_true', default=False,
                        help='display option')
    parser.add_argument('--display', action='store_true', default=False,
                        help='display option')



    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--pca', action='store_true', default= True)
    parser.add_argument('--confusion_matrix', action='store_true', default=True)
    parser.add_argument('--accuracy_values', action='store_true', default=True)


    global args, device
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)


    main()