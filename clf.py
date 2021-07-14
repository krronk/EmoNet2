from torchvision import models, transforms
import torch
from PIL import Image

def predict(image_path):
    resnet = torch.load()

    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(120),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.4681, 0.4030, 0.5275] , std =[1.5016, 1.5685, 1.5956])])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open('remonet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
