import _init_paths
import os
import argparse
import pickle
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import cv2
from model import net, embedding

from utils.gen_utils import make_dir_if_not_exist

from config.base_config import cfg, cfg_from_file
from sklearn.decomposition import PCA

import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import datasets
from torch.utils.data import Dataset
means = (0.485,)
stds = (0.229,)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds)])



class TestingDataset(Dataset):
    def __init__(self, annotations_file,  img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,  self.img_labels.iloc[idx, 2], self.img_labels.iloc[idx, 0])

        image = cv2.imread(img_path)
        print (img_path)
        image = cv2.resize(image, (228, 228))
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        return image, label






def main():
    torch.cuda.set_device(args.gpu_devices)

    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    exp_dir = os.path.join("data", args.exp_name)
    make_dir_if_not_exist(exp_dir)



    if args.pkl is not None:
        input_file = open(args.pkl, 'rb')
        final_data = pickle.load(input_file)
        input_file.close()
        embeddings = final_data['embeddings']
        labels = final_data['labels']
        vis_tSNE(embeddings, labels)       #visualize embeddings
    #else loop only works
    else:
        embeddingNet = None
        #if (args.dataset == 's2s') or (args.dataset == 'vggface2') or (args.dataset == 'custom'):
        #    embeddingNet = embedding.EmbeddingResnet()
        #elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
        #    embeddingNet = embedding.EmbeddingLeNet()
        if (args.dataset == 'cleft_lip_alexnet'):
            embeddingNet = embedding.EmbeddingAlexNet()
        elif (args.dataset == 'cleft_lip_resnet'):
            embeddingNet = embedding.EmbeddingResnet()
        else:
            print("Dataset {} not supported ".format(args.dataset))
            return

        model_dict = None
        if args.ckp is not None:
            if os.path.isfile(args.ckp):
                print("=> Loading checkpoint '{}'".format(args.ckp))
                try:
                    model_dict = torch.load(args.ckp)['state_dict']
                except Exception:
                    model_dict = torch.load(args.ckp, map_location='cpu')['state_dict']
                print("=> Loaded checkpoint '{}'".format(args.ckp))
            else:
                print("=> No checkpoint found at '{}'".format(args.ckp))
                return
        else:
            print("Please specify a model")
            return

        model_dict_mod = {}
        for key, value in model_dict.items():
            new_key = '.'.join(key.split('.')[2:])
            model_dict_mod[new_key] = value
        model = embeddingNet.to(device)
        model.load_state_dict(model_dict_mod, strict=False)

        data_loader = None
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        #if (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
        #    transform = transforms.Compose([
        #        transforms.ToPILImage(),
        #        transforms.Normalize((0.1307,), (0.3081,)),
        #        transforms.ToTensor()
        #    ])
        #    train_dataset = None
            #if args.dataset == 'mnist':
            #    train_dataset = MNIST('data/MNIST', train=True, download=True, transform=transform)
            #if args.dataset == 'fmnist':
            #    train_dataset = FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transform)
            #data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        if args.dataset == 'custom' or args.dataset == 'cleft_lip_alexnet'or args.dataset == 'cleft_lip_resnet':

            train_dataset = None
            means = (0.485,)
            stds = (0.229,)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds)
            ])
            train_dataset = TestingDataset(args.annot_file, args.data_dir, transform)
            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        else:
            print("Dataset {} not supported ".format(args.dataset))
            return
        #siamese dataloader will be used here
        #have to read files and add to dataloader


        embeddings, labels = generate_embeddings(data_loader, model)
        #embeddings = ndarray(596, 9216) in float, labels =ndarray(596)
        final_data = {
            'embeddings': embeddings,
            'labels': labels
        }
        matrix = "pca"
        if matrix == "tsne":
            dst_dir = os.path.join('data', args.exp_name, 'tSNE')
            make_dir_if_not_exist(dst_dir)

            output_file = open(os.path.join(dst_dir, 'tSNE.pkl'), 'wb')
            pickle.dump(final_data, output_file)
            output_file.close()

            vis_tSNE(embeddings, labels)
        elif matrix == "pca":
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
            plt.savefig('plot_PCA.png')
            plt.show()




def generate_embeddings(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = None
        embeddings = None
        inc = 0
        for batch_idx, data in tqdm(enumerate(data_loader)):
            print (inc)
            inc +=1
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            batch_imgs = Variable(batch_imgs.to(device))
            bacth_E = model(batch_imgs)
            bacth_E = bacth_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, bacth_E), axis=0) if embeddings is not None else bacth_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels


def vis_tSNE(embeddings, labels):
    num_samples = args.tSNE_ns if args.tSNE_ns < embeddings.shape[0] else embeddings.shape[0]
    X_embedded = TSNE(n_components=2).fit_transform(embeddings[0:num_samples, :])

    fig, ax = plt.subplots()

    x, y = X_embedded[:, 0], X_embedded[:, 1]
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    sc = ax.scatter(x, y, c=labels[0:num_samples], cmap=mpl.colors.ListedColormap(colors))
    plt.colorbar(sc)
    #plt.savefig(os.path.join('data', args.exp_name, 'tSNE', 'tSNE_' + str(num_samples) + '.jpg'))
    plt.savefig('/home/daniel/simase_network/pytorch-siamese-triplet/data/cleft_alexnet/tsne_res.jpg')
    print ('saved_file')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--exp_name', default='alexnet_triplet_2023', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--ckp', default='/home/daniel/simase_network/pytorch-siamese-triplet/data/resnet_triplet_2023/checkpoint_1.pth', type=str,
                        help='path to load checkpoint')
    parser.add_argument('--annot_file',
                        default='/home/daniel/simase_network/Cleft_lip_data/csv_files/test_file.csv',
                        type=str,
                        help='path to annotation file')
    parser.add_argument('--data_dir',
                        default='/home/daniel/simase_network/Cleft_lip_data/test',
                        type=str,
                        help='path to data Directory')
    parser.add_argument('--dataset', type=str, default='cleft_lip_resnet', metavar='M',
                        help='Dataset name')

    parser.add_argument('--pkl', default=None, type=str,
                        help='Path to load embeddings')

    parser.add_argument('--tSNE_ns', default=596, type=int,
                        help='Num samples to create a tSNE visualisation')
    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None,
                        help="List of GPU Devices to train on")
    global args, device
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    cfg_from_file("config/test.yaml")

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = 1
    else:
        device = 'cpu'
    main()
