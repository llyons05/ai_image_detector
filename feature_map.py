import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random
import math
import os

from image_identifier import Image_Identifier
import data_loader as dl


def get_layers(model: Image_Identifier) -> nn.Module:
    return model.conv


def show_maps():
    model: Image_Identifier = dl.load_existing_model_state("best_model.pth")[0]
    model_layers = get_layers(model)
    image = load_image()

    model.eval()
    with torch.no_grad():
        print(nn.functional.sigmoid(model(image.view(1, 3, 32, 32)))[0][0])

    outputs = [[image]]
    names = [["base image"]]
    for i, layer in enumerate(model_layers.children()):
        layer.eval()
        if type(layer) == nn.Dropout2d:
            continue

        if type(layer) == nn.Conv2d:
            layer_outputs = []
            layer_names = []
            for j in range(layer.weight.data.shape[0]):
                filter_j = layer.weight.data[j, :, :, :].unsqueeze(0)
                output_map = nn.functional.conv2d(image, filter_j, stride=1, padding=1)
                layer_outputs.append(output_map)
                layer_names.append(str(j))

            image = layer(image)
            layer_outputs.append(image)
            layer_names.append("FINAL")

            outputs.append(layer_outputs)
            names.append(layer_names)

        else:
            image = layer(image)
    
    for i in range(len(outputs)):
        save_outputs(outputs[i], names[i], f"{dl.FIGURE_DIR}/feature_map_layer_{i}.jpg")


def save_outputs(outputs: list[torch.Tensor], names: list[str], filename: str):
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    cols = 10
    rows = math.ceil(len(outputs)/cols)

    
    fig = plt.figure(figsize=(cols*2, rows*2))
    for i in range(len(processed)):
        a = fig.add_subplot(rows, cols, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=cols*.75)
    plt.savefig(filename, bbox_inches='tight')


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
    dl.ensure_all_dirs_exist()
    show_maps()


if __name__ == "__main__":
    torch.set_default_device("cpu")
    main()