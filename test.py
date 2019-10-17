from dataset import img_dataset
import torch.nn as nn
import torch
import argparse
import torchvision
import torch.optim as optim
from os.path import join
from model import Classifier_2, Classifier
import csv

class_name = [
    "bedroom",
    "coast",
    "forest",
    "highway",
    "insidecity",
    "kitchen",
    "livingroom",
    "mountain",
    "office",
    "opencountry",
    "street",
    "suburb",
    "tallbuilding",
]


def test(config):
    dataset = img_dataset("./dataset/test", "test")
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=config.bs
    )

    net = torchvision.models.resnet50(num_classes=13).cuda()
    # net = torchvision.models.vgg19(num_classes=13).cuda()
    # net = Classifier_2(num_classes=13).cuda()
    net.load_state_dict(
        torch.load(
            join(
                f"./weights/{config.model}/",
                f"{config.test_epoch}_{config.model}.pth",
            )
        )
    )
    net.eval()
    with open("output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["id", "label"])
        for _, data in enumerate(dataloader):
            with torch.no_grad():
                inputs, img_name = data
                inputs = inputs.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                writer.writerow([img_name[0], class_name[predicted.data]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bs", default=1, type=int, help="training batch size"
    )
    parser.add_argument(
        "--test_epoch", default=1500, type=int, help="testing epochs"
    )
    parser.add_argument(
        "--model", default="resnet50_flip", type=str, help="testing model"
    )
    config = parser.parse_args()
    test(config)
