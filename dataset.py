from torch.utils import data
import os
from os import listdir
from os.path import join
from PIL import Image
from torchvision import transforms

class_name = {
    "bedroom": 0,
    "coast": 1,
    "forest": 2,
    "highway": 3,
    "insidecity": 4,
    "kitchen": 5,
    "livingroom": 6,
    "mountain": 7,
    "office": 8,
    "opencountry": 9,
    "street": 10,
    "suburb": 11,
    "tallbuilding": 12,
}


def getData(root, mode):
    if mode == "train":
        images_labels = []
        image_names = []
        for class_id in listdir(root):
            class_dir = join(root, class_id)
            if os.path.isdir(class_dir):
                for image in listdir(class_dir):
                    image_names.append(join(class_id, image))
                    images_labels.append(class_name[class_id])
        return image_names, images_labels
    else:
        image_names = []
        for image in listdir(root):
            image_names.append(image)
        return image_names


class img_dataset(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        if mode == "train":
            self.img_name, self.label = getData(root, mode)
        else:
            self.img_name = getData(root, mode)

        print("> Found {} images...".format(len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        image_path = join(self.root, self.img_name[index])
        img = Image.open(image_path)
        img = img.convert("RGB")
        if self.mode == "train":
            train_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                ]
            )
            img = train_transforms(img)
            label = self.label[index]

            return img, label

        else:
            test_transforms = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
            img = test_transforms(img)
            img_name = self.img_name[index].split(".")[0]
            return img, img_name
