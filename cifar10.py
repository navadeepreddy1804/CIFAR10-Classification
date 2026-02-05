import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from PIL import Image

device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_data=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_data=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)
classes=train_data.classes

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc=nn.Sequential(
            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    
    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=15
for epoch in range(epochs):
    model.train()
    total_loss=0
    for imgs,labels in train_loader:
        imgs,labels=imgs.to(device),labels.to(device)
        pred=model(imgs)
        loss=criterion(pred,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct=0
total=0
with torch.no_grad():
    for imgs,labels in test_loader:
        imgs,labels=imgs.to(device),labels.to(device)
        pred=model(imgs)
        _,predicted=torch.max(pred,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
print(f"Accuracy: {100*correct/total:.2f}%")

def pred_img(img_path):
    img=Image.open(img_path).convert('RGB')
    img=transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output=model(img)
        _,predicted=torch.max(output,1)
    class_name=classes[predicted.item()]
    print(class_name)
pred_img("t7.png")