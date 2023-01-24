import torch

from sklearn.decomposition import PCA
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

from model import net, embedding

from utils.gen_utils import make_dir_if_not_exist

from config.base_config import cfg, cfg_from_file

import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
from sklearn.manifold import TSNE
import umap
import umap.plot

from siamese_classes import twoimage_inference, SiameseNetwork101, twoimage_inference_features


def eucledian(out, distance_list, label_list):
    x1 = [];
    y1 = []
    x2 = [];
    y2 = []
    x3 = [];
    y3 = []
    count = 0
    for (distance, label) in zip(distance_list, label_list):
        if label == 0:
            x1.append(count)
            y1.append(distance)
        if label == 1:
            x2.append(count)
            y2.append(distance)
        if label == 2:
            x3.append(count)
            y3.append(distance)
        count += 1
    plt.scatter(x1, y1, c='b', marker='x', label='1')
    plt.scatter(x2, y2, c='r', marker='s', label='-1')
    plt.scatter(x3, y3, c='k', marker='s', label='-1')

    plt.xlabel("Images")
    plt.ylabel("Eucledian Distance")
    plt.savefig(working_path + 'AlexNet_scores/test_results/training_EUC_DIST_plot.png')


def pca(out, label_list, name_list):
    label_df = pd.DataFrame(label_list, columns=["label"])
    name_df = pd.DataFrame(name_list, columns=["name"])

    pca = PCA(n_components=2)
    principalcomponents = pca.fit_transform(out)
    pcadf = pd.DataFrame(principalcomponents, columns=['pc1', 'pc2'])
    pcadf = pd.concat([pcadf, label_df], axis=1)

    classlist_compare = pd.concat([pcadf, name_df], axis=1)
    pca_compare = pd.concat([pcadf, label_df], axis=1)
    classlist_compare.to_csv("a_file.csv")

    plt.figure(figsize=(16, 9))
    groups = pcadf.groupby('label')
    for name, group in groups:
        plt.plot(group.pc1, group.pc2, marker='o', linestyle='', markersize=10, label=name)
    plt.legend()
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.savefig(working_path + 'AlexNet_scores/test_results/plot_PCA_testing.png')
    print(pcadf)


def tsne(out_features, label_list):
    x = np.array(out)
    y = np.array(label_list)
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]
    seaborn.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=seaborn.color_palette("hls", 3),
                        data=df).set(title="Iris data T-SNE projection")

    plt.savefig(working_path + 'AlexNet_scores/test_results/tsne_validation.png')


def umap_redn(out_features, label_list):
    x = np.array(out)
    y = np.array(label_list)
    mapper = umap.UMAP().fit(x)
    umap.plot.points(mapper, labels=y)

    plt.savefig(working_path + 'AlexNet_scores/test_results/UMAP_training.png')


working_path = '/home/daniel/simase_network/SiameseChange/'

net = SiameseNetwork101().cuda()
net.load_state_dict(
    torch.load('/home/daniel/simase_network/SiameseChange/AlexNet_scores/siamese_ROP_model_good_epoch19.pth'))

test_file = working_path + 'images_scores_900/training_files/training.txt'
folder = working_path + 'images_scores_900/'

with open(test_file, 'r') as f:
    lines = f.readlines()

index = 0
print(index)
healthy_dict = {}
cleft_dict = {}
label_list = []
name_list = []
distance_list = []
for j, name in enumerate(lines):
    img_location1 = folder + lines[index].strip()
    img_location2 = folder + name.strip()

    out1, out2 = twoimage_inference_features(img_location1, img_location2, net)

    distance = twoimage_inference(img_location1, img_location2, net)
    print(type(distance_list))
    distance_list.append(distance)

    # print (out2.cpu().detach().numpy().shape)
    classs = int(name.split('_')[-1].split('.')[0])

    try:
        out = np.append(out, out2.cpu().detach().numpy(), axis=0)
    except NameError:
        out = out2.cpu().detach().numpy()
    print(out.shape)
    if classs == 0:
        label_list.append(0)
    elif classs == 1 or classs == 2:
        label_list.append(1)
    elif classs == 3 or classs == 4 or classs == 5:
        label_list.append(2)
    name_list.append(name.strip())

# label_df = pd.DataFrame(label_list, columns=["label"])
# name_df = pd.DataFrame(name_list, columns=["name"])

# eucledian(out, distance_list, label_list)
# pca(out, label_list, name_list)
tsne(out, label_list)
#umap_redn(out, label_list)



def main():
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
        tsne(embeddings, labels)
    else:
        embeddingNet = None
        if (args.dataset == 's2s') or (args.dataset == 'vggface2') or (args.dataset == 'custom'):
            embeddingNet = embedding.EmbeddingResnet()
        elif (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
            embeddingNet = embedding.EmbeddingLeNet()
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
        model.load_state_dict(model_dict_mod)

        data_loader = None
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        if (args.dataset == 'mnist') or (args.dataset == 'fmnist'):
            transform = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = None
            if args.dataset == 'mnist':
                train_dataset = MNIST('data/MNIST', train=True, download=True, transform=transform)
            if args.dataset == 'fmnist':
                train_dataset = FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transform)
            data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
        else:
            print("Dataset {} not supported ".format(args.dataset))
            return
        #siamese dataloader will be used here
        #have to read files and add to dataloader


        embeddings, labels = generate_embeddings(data_loader, model)

        final_data = {
            'embeddings': embeddings,
            'labels': labels
        }

        dst_dir = os.path.join('data', args.exp_name, 'tSNE')
        make_dir_if_not_exist(dst_dir)

        output_file = open(os.path.join(dst_dir, 'tSNE.pkl'), 'wb')
        pickle.dump(final_data, output_file)
        output_file.close()

        vis_tSNE(embeddings, labels)


def generate_embeddings(data_loader, model):
    with torch.no_grad():
        model.eval()
        labels = None
        embeddings = None
        for batch_idx, data in tqdm(enumerate(data_loader)):
            batch_imgs, batch_labels = data
            batch_labels = batch_labels.numpy()
            batch_imgs = Variable(batch_imgs.to(device))
            bacth_E = model(batch_imgs)
            bacth_E = bacth_E.data.cpu().numpy()
            embeddings = np.concatenate((embeddings, bacth_E), axis=0) if embeddings is not None else bacth_E
            labels = np.concatenate((labels, batch_labels), axis=0) if labels is not None else batch_labels
    return embeddings, labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Siamese Example')
    parser.add_argument('--exp_name', default='cleft_alexnet', type=str,
                        help='name of experiment')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--ckp', default='/home/daniel/simase_network/pytorch-siamese-triplet/data/cleft_alexnet/checkpoint_9.pth', type=str,
                        help='path to load checkpoint')
    parser.add_argument('--dataset', type=str, default='custom', metavar='M',
                        help='Dataset (default: mnist)')

    parser.add_argument('--pkl', default=None, type=str,
                        help='Path to load embeddings')

    parser.add_argument('--tSNE_ns', default=98, type=int,
                        help='Num samples to create a tSNE visualisation')

    global args, device
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    cfg_from_file("config/test.yaml")

    if args.cuda:
        device = 'cuda'
        if args.gpu_devices is None:
            args.gpu_devices = [0]
    else:
        device = 'cpu'
    main()
'''
id1 = 17
healthy = [3, 4, 7, 8, 10, 13, 18, 19, 20, 21, 22, 23]

cleft = [0, 1, 2, 6, 5, 9, 11, 12, 14, 15, 16, 17]
dicth = {}
dictc = {}
for k in healthy:
    image_location1 = folder + lines[id1].split('.')[0] + '.png'
    image_location2 = folder + lines[k].split('.')[0] + '.png'

    out = twoimage_inference(image_location1, image_location2, net)

    print (out, image_location1.split('_')[-1].split('.')[0], image_location2.split('_')[-1].split('.')[0])
    dicth[k] = out


for k in cleft:
    image_location1 = folder + lines[id1].split('.')[0] + '.png'
    image_location2 = folder + lines[k].split('.')[0] + '.png'

    out = twoimage_inference(image_location1, image_location2, net)

    print (out, image_location1.split('_')[-1].split('.')[0], image_location2.split('_')[-1].split('.')[0])
    dictc[k] = out

print (dicth, dictc)

print ()
x1 = list(dicth.keys())
y1 = list(dicth.values())
x2 = list(dictc.keys())
y2 = list(dictc.values())

plt.scatter(x1,y1, c='b', marker='x', label='1')
plt.scatter(x2, y2, c='r', marker='s', label='-1')
plt.savefig('plot.png')
'''