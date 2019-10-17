import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_classes=13):
        super(Classifier, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(
                256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
            nn.Conv2d(
                512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
            ),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=25088, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=2048, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
