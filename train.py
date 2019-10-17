from dataset import img_dataset
import torch.nn as nn
import torch
import argparse
import torchvision
import torch.optim as optim
from os.path import join
import lera
from model import Classifier_2
def train(config):
    lera.log_hyperparams({
        'title':'hw1',
        'model':config.model,
        'epoch':config.epochs,
        'lr':config.lr,
    })
    dataset = img_dataset('./dataset/train', 'train')
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=config.bs,
                                                    shuffle=True,
                                                    drop_last=True)

    # net = torchvision.models.resnet50(num_classes=13).cuda()
    #net = torchvision.models.vgg19(num_classes=13).cuda()
    net = Classifier_2(num_classes=13).cuda()
    net.load_state_dict(torch.load(join(f'./weights/{config.model}/', f'{config.pre_epochs}_{config.model}.pth')))
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    for epoch in range(config.epochs):  
        for _, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            net.train()
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            _, predicted = torch.max(outputs.data, 1)
            correct_counts = predicted.eq(labels.data.view_as(predicted))
            train_acc = torch.sum(correct_counts).item()/predicted.size(0)
        lera.log({
            "loss":loss.item(),
            "acc": train_acc
        })
        print("epoch:{}/{}, loss:{}, acc:{:02f}".format(epoch + 1 + config.pre_epochs, config.epochs + config.pre_epochs, loss.item(), train_acc))
        if (epoch + 1 + config.pre_epochs) % 10 == 0:
            torch.save(net.state_dict(), join(f'./weights/{config.model}/', f'{epoch + 1 + config.pre_epochs}_{config.model}.pth'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", default=64, type=int, help='training batch size')
    parser.add_argument("--lr", default=0.00005, type=float, help='training learning rate')
    parser.add_argument("--epochs", default=2000, type=int, help='training epochs')
    parser.add_argument("--pre_epochs", default=3000, type=int, help='training epochs')
    parser.add_argument("--model", default='classifier_2', type=str, help='training model')
    config = parser.parse_args()
    train(config)
   
     