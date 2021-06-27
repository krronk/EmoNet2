

from PIL import Image
import torch , torchvision
from torchvision import models, transforms
import streamlit as st

# set title of app
st.title("Emotion Classification App")
st.write(" Emotion recognition according to seven basic emotions")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["jpg" , "jpeg" , "png"])


def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    device = torch.device('cpu')
    mnet = torch.load('mnot_1_68.pth' , map_location= device)
    mnet.eval()


    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(120),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4681, 0.4030, 0.5275],
            std = [1.5016, 1.5685, 1.5956]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    out = mnet(batch_t)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

    with open('emonet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        
     
    top5_prob, top5_catid = torch.topk(probabilities*100, 5)
    for i in range(top5_prob.size(0)):
        
        return (classes[top5_catid[i]], top5_prob[i].item())


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)
    st.write(labels)

    # print out the top 5 prediction labels with scores
    #for i in labels:
    #    st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
