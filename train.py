import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import swanlab

# 1. 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    # 2. 超参数设置
    config = {
        "learning_rate": 0.01,
        "batch_size": 64,
        "epochs": 3,
        "dataset": "MNIST",
        "architecture": "CNN"
    }

    # 3. 初始化 SwanLab（这会在云端为你创建一个项目）
    swanlab.init(
        project="Intern-Application",
        experiment_name="My-First-CV-Model",
        config=config,
        description="后端实习生申请：体验核心训练流与数据上报API"
    )

    # 4. 数据加载
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # 硬件加速：检查Mac的MPS或使用CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 5. 初始化模型、损失函数和优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

    # 6. 训练循环
    model.train()
    for epoch in range(1, config["epochs"] + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device) # 数据放入设备
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 每 100 个 batch 记录一次数据到 SwanLab 云端
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
                # 【核心】：使用 log 方法上传标量和图像
                swanlab.log({
                    "train_loss": loss.item(),
                    # 抽取第一张图片传到云端，注意把 tensor 放回 cpu 以便图像处理
                    "sample_image": swanlab.Image(data[0].cpu(), caption=f"Label: {target[0].item()}") 
                })

if __name__ == "__main__":
    main()
