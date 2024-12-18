# Importing dependencies
import streamlit as st
from PIL import Image
import torch
import torchvision.models as models
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm


st.title("PyTorch Neural Style Transfer :smile:")
st.title("By : Manas Ranjan Jena.")

img = st.sidebar.selectbox(
    'Select Image',
    ['Cat.png', 'Face.jpg', 'Scenery.jpg', 'Dog.png'])

style_name = st.sidebar.selectbox('Select Style',
                                  ['Style_1', 'Style_2', 'Style_3'])

input_image = 'Neural_Style_Transfer_Model/Images/' + img
output_img = 'Neural_Style_Transfer_Model/Output_Image/' + style_name + '-' + img

st.write('### Source Image :')
image = Image.open(input_image)
st.image(image, width = 400)
for k in range(1, 4):
    st.write(f"### NST_Style_{k}")
    style = Image.open(f"Neural_Style_Transfer_Model/Styles/Style {k}.png" )
    st.image(style, width = 150)

clicked = st.button('Stylize')

model = models.vgg19(pretrained=True).features
feature_index = [0, 5 , 10, 19, 28]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 256
loader = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor()])
def load_image(image_path):
    """
    Function to load an image from the specified file path.
    :param image_path: str
    :return: torch.Tensor
    """
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device)

origin_img = load_image(input_image)

if clicked:
    st.write('### Stylising the image............')
    st.write('It may take 3-5 minutes for the image to be generated.')
    model1_state_dict = torch.load('Neural_Style_Transfer_Model/Models/NST_Style_1.pth', weights_only = True, map_location = device)
    model2_state_dict = torch.load('Neural_Style_Transfer_Model/Models/NST_Style_2.pth', weights_only = True, map_location = device)
    model3_state_dict = torch.load('Neural_Style_Transfer_Model/Models/NST_Style_3.pth', weights_only = True, map_location = device)

    class vgg(nn.Module):
        def __init__(self):
            super().__init__()
            self.chosen_features = feature_index
            self.model = models.vgg19(pretrained=True).features[:29]   # Taking the features from the pretrained vgg 19 model upto 29 indexing as we need features only upto 28 indexing for our model.

        def forward(self, x):
            features = []
            for layer_num, layer in enumerate(self.model):   # layer_num will be containing the index of the layer currently being iterated and layer will contain the convolution layer at that indexing.
                x = layer(x)
                if layer_num in self.chosen_features:
                    features.append(x)   # Collecting the value of only after it is passed through the choosen feature/convolution layer.
            return features


    model = vgg().to(device)
    model.eval()

    if style_name == 'Style_1':
        style = load_image('Neural_Style_Transfer_Model/Styles/Style 1.png')
        loaded_model = model.load_state_dict(model1_state_dict)
    elif style_name == 'Style_2':
        style = load_image('Neural_Style_Transfer_Model/Styles/Style 2.png')
        loaded_model = model.load_state_dict(model2_state_dict)
    elif style_name == 'Style_3':
        style = load_image('Neural_Style_Transfer_Model/Styles/Style 3.png')
        loaded_model = model.load_state_dict(model3_state_dict)
    generated_image = origin_img.clone().requires_grad_(True)
    # Requires grad specifies that the gradient descent or the optimisation will be done on the generated_image.

    # Setting The hyperparameters
    epochs = 10
    lr = 0.1
    alpha = 1.5     # To be multiplied with the content loss.
    beta = 0.001      # To be multiplied with the style loss.
    # The alpha and beta determines how much of the structure from the original image or how much style do we need in the generated image.
    optimizer = torch.optim.Adam([generated_image], lr = lr)    # Generally we take model.params() as argument but here we need to freeze the model weights so, the loss optimisation will be done on the generated image rather than the model parameters.

    torch.manual_seed(21)
    # Loop for training our model
    for epoch in tqdm(range(epochs)):
        generated_features = model(generated_image)
        original_img_features = model(origin_img)
        style_features = model(style)
        style_loss, original_loss = 0, 0
        for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
            batch_size, channel, height, width = gen_feature.shape
            original_loss = original_loss + torch.mean((gen_feature - orig_feature)**2)    # Calculating the mean squared error.

            # Computing the gram matrix
            G = gen_feature.view(channel, height*width).mm(gen_feature.view(channel, height*width).t())   # Here mm means matrix multiplication with self and the matrix passed inside it and t means transpose of self.
            # Here we are not multiplying the batch because batch is 1.
            A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t())
            style_loss = style_loss + torch.mean((G - A)**2)
        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print("Total loss = ", total_loss)
    save_image(generated_image, output_img)

    st.write('### Your Stylised Image :smile:')
    st.image(output_img)










