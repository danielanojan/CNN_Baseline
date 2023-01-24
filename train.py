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

wandb.init(project="vgg19-cnn",

config={
    "learning_rate": 0.0001,
    "architecture": "vgg19",
    "dataset": "596_img_cleftlip",
    "epochs": 20,
    }
)

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





#siamese dataloader will be used here
#have to read files and add to dataloader


def main():
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)
    #cudnn.benchmark = True #optimizes when the input size of the network is same. dont use when the input sizes are different.

    exp_dir = os.path.join(args.result_dir, args.exp_name)
    make_dir_if_not_exist(exp_dir)

    #dataloader classes and the transforms are defined here. More transforms should be tried

    train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
        
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = DatasetClass(args.test_annot_file, args.test_data_dir, test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataset = DatasetClass(args.train_annot_file, args.train_data_dir, train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Build Model
    model = get_model(args, device)
    if model is None:
        return

    # Criterion and Optimizer
    #params = []
    #for key, value in dict(model.named_parameters()).items():
    #    if value.requires_grad:
    #        params += [{'params': [value]}]

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = optim.SGD(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    #optimizer = optim.Adam(params, lr=args.lr)
    best_acc = 0.0
    # Train Test Loop
    for epoch in range(1, args.epochs + 1):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            # Init data loaders
            #train_data_loader, test_data_loader = get_loader(args)
            # Test train

            train_epoch_loss, train_epoch_acc = train(train_data_loader, model, criterion, optimizer_ft, epoch, running_loss, exp_lr_scheduler)
            test_epoch_loss, test_epoch_acc = test(test_data_loader, model, criterion, optimizer_ft, epoch, running_loss)

            #test_loss, accuracy_zero_margin = test(test_data_loader, model, criterion)
            test_dataset_emb = DatasetClass(args.test_annot_file, args.test_data_dir, test_transform)
            test_data_loader_emb = torch.utils.data.DataLoader(test_dataset_emb, batch_size=1, shuffle=False)

            train_dataset_emb = DatasetClass(args.train_annot_file, args.train_data_dir, train_transform)
            train_data_loader_emb = torch.utils.data.DataLoader(train_dataset_emb, batch_size=1, shuffle=True)

            embedding_dl_train = test_data_loader_emb
            embeddings_train, labels_train = generate_emb(embedding_dl_train, model)
            pca_plot(embeddings_train, labels_train, epoch, label=0000)

            embedding_dl_test = train_data_loader_emb
            embeddings_test, labels_test = generate_emb(embedding_dl_test, model)
            pca_plot(embeddings_test, labels_test, epoch, label=11111)

            # Save model
            model_to_save = {
                "epoch": epoch + 1,
                'state_dict': model.state_dict(),
            }
            if epoch % args.ckp_freq == 0:
                file_name = os.path.join(exp_dir, "checkpoint_" + str(epoch) + ".pth")
                save_checkpoint(model_to_save, file_name)
            wandb.log({'train_loss': train_epoch_loss, 'train_acc': train_epoch_acc,  'test_loss': test_epoch_loss, 'test_acc': test_epoch_acc })

def train(data, model, criterion, optimizer, epoch, train_running_loss, scheduler):
    phase = 'train'
    print("******** Training ********")
    running_corrects = 0

    for inputs, labels in data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.to(memory_format=torch.contiguous_format)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            print (loss)
            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()

        # statistics
        train_running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        #log_step = args.train_log_step
        #if (batch_idx % log_step == 0) and (batch_idx != 0):
        #    print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
        #    total_loss = 0

    if phase == 'train':
        scheduler.step()
    train_epoch_loss = train_running_loss / len(data.dataset)
    train_epoch_acc = running_corrects.double() / len(data.dataset)
    print('{}  {} Loss: {:.4f} Acc: {:.4f}'.format(epoch,
        phase, train_epoch_loss, train_epoch_acc))
    return train_epoch_loss, train_epoch_acc

def test(data, model, criterion, optimizer, epoch, test_running_loss):
    phase = 'test'
    print("******** Training ********")
    running_corrects = 0

    for inputs, labels in data:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.to(memory_format=torch.contiguous_format)
        # zero the parameter gradients
        optimizer.zero_grad()
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
        # statistics
        test_running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        # log_step = args.train_log_step
        # if (batch_idx % log_step == 0) and (batch_idx != 0):
        #    print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
        #    total_loss = 0
    test_epoch_loss = test_running_loss / len(data.dataset)
    test_epoch_acc = running_corrects.double() / len(data.dataset)
    print('{}  {} Loss: {:.4f} Acc: {:.4f}'.format(epoch,
                                                   phase, test_epoch_loss, test_epoch_acc))

    return test_epoch_loss, test_epoch_acc


    '''
    for batch_idx, img_triplet in enumerate(data):
        anchor_img, pos_img, neg_img = img_triplet
        anchor_img, pos_img, neg_img = anchor_img.to(device), pos_img.to(device), neg_img.to(device)
        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
        E1, E2, E3 = model(anchor_img, pos_img, neg_img)
        dist_E1_E2 = F.pairwise_distance(E1, E2, 2)
        dist_E1_E3 = F.pairwise_distance(E1, E3, 2)

        target = torch.FloatTensor(dist_E1_E2.size()).fill_(-1)
        target = target.to(device)
        target = Variable(target)
        loss = criterion(dist_E1_E2, dist_E1_E3, target)
        total_loss += loss
        log_epoch_loss +=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_step = args.train_log_step
        if (batch_idx % log_step == 0) and (batch_idx != 0):
            print('Train Epoch: {} [{}/{}] \t Loss: {:.4f}'.format(epoch, batch_idx, len(data), total_loss / log_step))
            total_loss = 0
    print("****************")
    epoch_loss = log_epoch_loss/ len(data)
    print ("EEEEEEEEEPOCH",epoch_loss)

    return epoch_loss
    '''

def save_checkpoint(state, file_name):
    torch.save(state, file_name)

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
    parser.add_argument('--exp_name', default='cnn_benchmark_vgg16', type=str,
                        help='name of experiment')
    parser.add_argument('--dataset', type=str, default='cleft_lip_vgg16', metavar='M',
                        help='Dataset')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None,
                        help="List of GPU Devices to train on")
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--ckp_freq', type=int, default=1, metavar='N',
                        help='Checkpoint Frequency (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--margin', type=float, default=1.0, metavar='M',
                        help='margin for triplet loss (default: 1.0)')
    parser.add_argument('--ckp', default=None, type=str,
                        help='path to load checkpoint')
    parser.add_argument('--num_train_samples', type=int, default=596, metavar='M',
                        help='number of training samples (default: 3000)')
    parser.add_argument('--num_test_samples', type=int, default=98, metavar='M',
                        help='number of test samples (default: 1000)')
    parser.add_argument('--train_log_step', type=int, default=1, metavar='M',
                        help='Number of iterations after which to log the loss')
    parser.add_argument('--train_annot_file',
                        default='/home/daniel/simase_network/Cleft_lip_data/csv_files/train_file.csv',
                        type=str,
                        help='path to annotation file')
    parser.add_argument('--train_data_dir',
                        default='/home/daniel/simase_network/Cleft_lip_data/train',
                        type=str,help='path to data Directory')
    parser.add_argument('--test_annot_file',
                        default='/home/daniel/simase_network/Cleft_lip_data/csv_files/test_file.csv',
                        type=str,
                        help='path to annotation file')
    parser.add_argument('--test_data_dir',
                        default='/home/daniel/simase_network/Cleft_lip_data/test',
                        type=str, help='path to data Directory')

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
