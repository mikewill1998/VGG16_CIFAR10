import torch
import torchvision
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets  
import torchvision.transforms as transforms
from torchvision.models import VGG16_BN_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
learning_rate = 0.001
batch_size = 512
num_epochs = 20

# Simple Identity class that let's input pass without changes
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load pretrain model & modify it
pretrained_weight = VGG16_BN_Weights.IMAGENET1K_V1
model = torchvision.models.vgg16_bn(weights=pretrained_weight)

# do finetuning then set requires_grad = False
# Remove these two lines if want to train entire model,
# and only want to load the pretrain weights.
# for param in model.parameters():
#     param.requires_grad = False

# model.avgpool = Identity()
model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=False),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=False),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=10, bias=False))


model.to(device)

# Load Data
tfm = transforms.Compose([transforms.ToTensor(), 
                          transforms.Normalize((0.49139968, 0.48215827, 0.44653124), 
                                               (0.24703233, 0.24348505, 0.26158768))])
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=tfm)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=tfm)
val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size*2, shuffle=False)
# Loss and optimizer
criterion = nn.CrossEntropyLoss() # F.cross_entropy also can
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# data, targets = next(iter(train_loader))

# Train
for epoch in range(num_epochs):
    model.train()
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
    # send data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # set gradient back to zero
        optimizer.zero_grad()

    print(f"Epoch [{epoch}], Loss {sum(losses)/len(losses):.4f}")

# Check accuracy
def validate_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
      # send data to cuda if possible
        for data, label in loader:
            data = data.to(device=device)
            label = label.to(device=device)

            scores = model(data)
            _, pred = scores.max(dim=1)
            num_correct += (pred == label).sum() # torch.sum(pred == v_label).item() also can
            num_samples += pred.size(0)

        print(f"Accuracy: {float(num_correct)/float(num_samples):.4f}")

    model.train()


validate_accuracy(train_loader, model)

validate_accuracy(val_loader, model)

torch.save(model.state_dict(), 'cifar10_vgg16.pth')