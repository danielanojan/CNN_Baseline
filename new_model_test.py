#script to test and implement from urllib.request import urlopen
#https://huggingface.co/timm/maxvit_tiny_tf_512.in1k
from urllib.request import urlopen
from PIL import Image
import timm
import torch

img = Image.open(
    urlopen('https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'))

model = timm.create_model('maxvit_tiny_tf_512.in1k', pretrained=True)
model = model.eval()
print ( model)
# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
img_b4 = img.copy()
img_mid = transforms(img)
img_aftr = transforms(img).unsqueeze(0)
output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)