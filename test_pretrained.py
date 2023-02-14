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
from model.net_pretrained import get_model, NeuralNet
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
class DatasetClass(Dataset):
    def __init__(self, annotations_file,  img_dir, args, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file)



    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,  self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 0])

        image = cv2.imread(img_path)
        #print (img_path)

        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    # cudnn.benchmark = True #optimizes when the input size of the network is same. dont use when the input sizes are different.

    exp_dir = os.path.join(args.result_dir, args.exp_name)
    make_dir_if_not_exist(exp_dir)

    current_date = datetime.date.today()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Build Model
    model = get_model(args, device)
    if model is None:
        return
    print (model)
    # model = NeuralNet()


    # test_loss, accuracy_zero_margin = test(test_data_loader, model, criterion)
    dataset_emb = DatasetClass(args.annot_file, args.data_dir, args, transform)
    data_loader_emb = torch.utils.data.DataLoader(dataset_emb, batch_size=1, shuffle=False)

    embedding_dl_train = data_loader_emb
    embeddings, labels = generate_emb(embedding_dl_train, model)
    if args.pca == True:
        epoch = 0
        pca_plot(embeddings, labels, epoch, label=0000)
    if args.confusion_matrix:
        predictions = np.argmax(embeddings, axis = 1)
        conf_matrix = confusion_matrix(labels, predictions, labels = [0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels= ['healthy','mild','severe'])
        disp.plot()
        #disp.xaxis.set_ticklabels(['healthy','mild','severe'])
        plt.show()
    if args.accuracy_values:
        running_corrects = 0
        for emb, label in zip(embeddings, labels.reshape(-1, 1)):
            pred = emb.argmax()
            if pred==label[0]:

                running_corrects += 1
        accuracy = running_corrects / len(data_loader_emb)
    print (accuracy)


def generate_emb(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = None
        embeddings = None
        inc = 0
        for batch_idx, data in tqdm(enumerate(data_loader)):
            #print (inc)
            inc +=1
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            batch_imgs = Variable(batch_imgs.to(device))
            bacth_E = model(batch_imgs)
            bacth_E = bacth_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, bacth_E), axis=0) if embeddings is not None else bacth_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels

def pca_plot(embeddings, labels, epoch, label):
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
        exp_dir = os.path.join(args.result_dir, args.exp_name)
        plt.savefig(os.path.join(exp_dir, 'plot_PCA{}_{}.png'.format(label, epoch)))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--result_dir', default='data', type=str,
                        help='Directory to store results')
    parser.add_argument('--exp_name', default='vgg_test', type=str,
                        help='name of experiment')
    parser.add_argument('--finetune',action='store_true', default=False,
                        help='freeze/unfreeze last layers')
    parser.add_argument('--archi', type=str, default='vgg16', metavar='M',
                        help='CNN Architectunre[vgg16, alexnet, resnet18, MaxxViT_tiny_512, mobilenet_v3_large, efficientnet_v2_large]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0,1],
                        help="List of GPU Devices to train on")
    parser.add_argument('--batch_size', type=int, default=0, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--pca', action='store_true', default= False)
    parser.add_argument('--confusion_matrix', action='store_true', default=True)
    parser.add_argument('--accuracy_values', action='store_true', default=True)
    parser.add_argument('--num_samples', type=int, default=98, metavar='M',
                        help='number of samples')
    parser.add_argument('--annot_file',
                        default='/mnt/recsys/daniel/simase_network/cleft_lip_data_600_800/csv_files/train_file.csv',
                        type=str,
                        help='path to annotation file')
    parser.add_argument('--data_dir',
                        default='/mnt/recsys/daniel/simase_network/cleft_lip_data_600_800/train',
                        type=str, help='path to data Directory')
    parser.add_argument('--ckp',
                        default='/mnt/recsys/daniel/simase_network/CNN_baseline/data/vgg_test/checkpoint_1.pth',
                        type=str,
                        help='path to pretrained file path')

    global args, device
    args = parser.parse_args()
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    args.cuda = args.cuda and torch.cuda.is_available()
    cfg_from_file("config/test.yaml")

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [1,2]
    else:
        device = 'cpu'
    main()