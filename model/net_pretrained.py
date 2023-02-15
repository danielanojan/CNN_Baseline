import os
import torch
import torch.nn as nn

from model import embedding
from torchvision import datasets, models, transforms

import timm

class Net(nn.Module):
    def __init__(self, embeddingNet):
        super(Net, self).__init__()
        self.embeddingNet = embeddingNet

    def forward(self, i1):
        E1 = self.embeddingNet(i1)
        return E1

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.vgg = models.vgg19(weights = 'IMAGENET1K_V1')
        #self.features = self.vgg.features
        #self.avgpool = self.vgg.avgpool
        #self.classifier = self.vgg.classifier
        #in_feat = self.classifier[6].in_features
        #self.classifier[6] =nn.Linear(in_features = in_feat, out_features = 3)
        #self.features = nn.Sequential(*list(self.features))
        #self.classifier = nn.Sequential(*list(self.classifier))
    def forward(self, x):
        out = self.vgg.features(x)
        #out = self.avgpool(out)
        #out = self.classifier(out)

def get_model(args, device):
    # Model
    embeddingNet = None
    if (args.archi == 'resnet18'):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_class)
        if args.finetune:
            for (name, module) in model.named_children():
                if name == 'conv1' or 'bn1'  or 'maxpool' or 'layer1' or 'layer2' or 'layer3' or 'layer4' or 'avgpool' :
                    for layer in module.children():
                        for param in layer.parameters():
                            if param.requires_grad:
                                param.requires_grad = False
    elif (args.archi == 'alexnet'):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        model.classifier[4] = nn.Linear(4096, 1024)
        model.classifier[6] = nn.Linear(1024, args.num_class)
        if args.finetune:
            for (name, module) in model.named_children():
                if name == 'features' or 'avgpool':
                    for layer in module.children():
                        for param in layer.parameters():
                            if param.requires_grad:
                                param.requires_grad = False
    elif (args.archi == 'vgg16'):
        model = models.vgg16(pretrained=True)

        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.num_class)
        if args.finetune:
            for (name, module) in model.named_children():
                if name == 'features' or 'avgpool':
                    for layer in module.children():
                        for param in layer.parameters():
                            if param.requires_grad:
                                param.requires_grad = False
    elif (args.archi == 'MaxxViT_tiny_512'):
        model = timm.create_model('maxvit_tiny_tf_512.in1k', pretrained=True)
        model.head.fc = nn.Linear(model.head.fc.in_features, args.num_class, bias=True)
        if args.finetune:
            for (name, module) in model.named_children():
                if name == 'stem' or 'stages' or 'norm':
                    for layer in module.children():
                        for param in layer.parameters():
                            if param.requires_grad:
                                param.requires_grad = False
    elif (args.archi == 'mobilenet_v3_large'):
        #acc@1 = 72.5 acc@5 = 92.5 trained on 232x232
        #https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html#torchvision.models.MobileNet_V3_Large_Weights
        model = models.mobilenet_v3_large(weights= 'IMAGENET1K_V2')
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=args.num_class, bias=True)
        if args.finetune:
            for (name, module) in model.named_children():
                if name == 'features' or 'avgpool':
                    for layer in module.children():
                        for param in layer.parameters():
                            if param.requires_grad:
                                param.requires_grad = False
    elif (args.archi == 'efficientnet_v2_large'):
        # acc@1 = 85.5 acc@5 = 97.7 trained on 232x232
        #https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_l.html#torchvision.models.efficientnet_v2_l
        model = models.efficientnet_v2_s(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_class, bias=True)
        if args.finetune:
            for (name, module) in model.named_children():
                if name == 'features' or 'avgpool':
                    for layer in module.children():
                        for param in layer.parameters():
                            if param.requires_grad:
                                param.requires_grad = False




    else:

        print("Architecture %s not supported " % args.archi)
        return None

    #print([n for n, _ in model.named_children()])
    ##########this can be changed?
    #model = Net(embeddingNet)
    #if args.cuda:
    #    model = nn.DataParallel(model, device_ids=args.gpu_devices)
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



