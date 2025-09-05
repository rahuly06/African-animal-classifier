import torch

class CNNModel(torch.nn.Module):
    def __init__(self, input_chanels):
        super(CNNModel, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(input_chanels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 56 * 56, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 5)

        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x