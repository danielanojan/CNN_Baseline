import os
import torch
import torch.nn as nn

from model import embedding
from torchvision import datasets, models, transforms


class Net(nn.Module):
    def __init__(self, embeddingNet):
        super(Net, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1):
        E1 = self.embeddingNet(i1)
        return E1


def get_model(args, device):
    # Model
    embeddingNet = None
    if (args.dataset == 'cleft_lip_resnet'):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif (args.dataset == 'cleft_lip_alexnet'):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model.classifier[4] = nn.Linear(4096, 1024)
        model.classifier[6] = nn.Linear(1024, 3)
    elif (args.dataset == 'cleft_lip_vgg16'):
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 3)
    else:
        print("Dataset %s not supported " % args.dataset)
        return None

    print([n for n, _ in model.named_children()])
    ##########this can be changed?
    #model = Net(embeddingNet)
    if args.cuda:
        model = nn.DataParallel(model, device_ids=args.gpu_devices)
    model = model.to(device)

    # Load weights if provided
    if args.ckp:
        if os.path.isfile(args.ckp):
            print("=> Loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint '{}'".format(args.ckp))
        else:
            print("=> No checkpoint found at '{}'".format(args.ckp))

    return model



