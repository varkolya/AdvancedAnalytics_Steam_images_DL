import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        # Feature extraction layers: Convolutional and pooling layers
        self.feature_extractor = nn.Sequential(
            # 3 input channels, 64 output channels, 3x3 kernel, 1 padding
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Max pooling with 2x2 kernel and stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            # 512 channels, 7x7 spatial dimensions after max pooling
            nn.Linear(512 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer with 0.5 dropout probability
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            # Output layer with 'num_classes' output units
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Pass input through the feature extractor layers
        x = self.feature_extractor(x)
        # print(x.shape)  # Print shape to debug
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        # print(x.shape)  # Print shape to debug
        # Pass flattened output through the classifier layers #(32x100352 and 25088x4096)
        x = self.classifier(x)
        return x