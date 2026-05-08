"""
任务二：LeNet-5 实现 - MNIST手写数字识别
经典结构适配：卷积+池化+全连接 标准组合
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载（与任务一保持一致，保证对比公平）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# LeNet-5 模型实现
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 输入：1x28x28
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 输出 6x28x28
        self.pool = nn.AvgPool2d(2, 2)                           # 池化
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 输出 16x10x10
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.sigmoid(self.conv1(x)))
        x = self.pool(torch.sigmoid(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练与测试逻辑
def train(epoch):
    model.train()
    loop = tqdm(train_loader, desc=f"LeNet5 训练轮次 {epoch}")
    for data, target in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100 * correct / total
    print(f"LeNet5 测试集准确率: {acc:.2f}%")
    return acc

# 统计训练时间
if __name__ == '__main__':
    print(f"使用设备: {device}")
    start_time = time.time()
    print("开始训练 LeNet-5...")
    for epoch in range(1, 6):
        train(epoch)
        test()
    total_time = time.time() - start_time
    print(f"总训练耗时: {total_time:.2f} 秒")
