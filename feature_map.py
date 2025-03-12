import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from image_identifier import Image_Identifier
from pathlib import Path
import random

def get_layers(model: Image_Identifier) -> nn.Module:
    return model.conv


def show_maps():
    model: Image_Identifier = torch.load("models/best_model.pth", weights_only=False).to(torch.get_default_device())
    model_layers = get_layers(model)
    image = load_image()

    model.eval()
    with torch.no_grad():
        print(nn.functional.sigmoid(model(image.view(1, 3, 32, 32)))[0][0])

    outputs = [image]
    names = ["base image"]
    for i, layer in enumerate(model_layers.children()):
        layer.eval()
        if type(layer) == nn.Dropout2d:
            continue

        if type(layer) == nn.Conv2d:
            for j in range(layer.weight.data.shape[0]):
                filter_j = layer.weight.data[j, :, :, :].unsqueeze(0)
                output_map = nn.functional.conv2d(image, filter_j, stride=1, padding=1)
                outputs.append(output_map)
                names.append(f"{i},{j}")

            image = layer(image)
            outputs.append(image)
            names.append(str(i))

        else:
            image = layer(image)
    
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    
    fig = plt.figure(figsize=(60, 100))
    for i in range(len(processed)):
        a = fig.add_subplot(20, 20, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
    # plt.ioff()
    # plt.show()



def load_image():
    main_dir = "datasets/test/"
    file_names = []
    transform = transforms.Compose([transforms.ToTensor()])

    for file in sorted((Path(main_dir).glob('*/*.*'))):
        file_names.append(file.absolute())

    filename = random.choice(file_names)
    print(filename)
    image = Image.open(filename)
    return transform(image).unsqueeze(0)


def main():
    show_maps()

if __name__ == "__main__":
    torch.set_default_device("cpu")
    main()