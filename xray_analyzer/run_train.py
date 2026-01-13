import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mashine_learning.training.trainer import train_model

# 1. Definicja klasy SimpleCNN (musi pasować do logów błędu!)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 32 kanały * 64x64 (po dwóch poolingach z 256x256)
        self.fc1 = nn.Linear(32 * 64 * 64, 128) 
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Inicjalizacja
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()

# 3. Dummy Data (do testu startu szkolenia)
X = torch.randn(10, 3, 256, 256)
y = torch.randint(0, 3, (10,))
train_loader = DataLoader(TensorDataset(X, y), batch_size=2)
val_loader = DataLoader(TensorDataset(X, y), batch_size=2)

# 4. Parametry
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == "__main__":
    print("Initializing model training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=6,
        save_dir="runs"
    )