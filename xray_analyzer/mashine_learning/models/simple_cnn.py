import torch
import torch.nn as nn
import torch.nn.functional as func

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = func.relu(self.fc1(x))
        return self.fc2(x)
