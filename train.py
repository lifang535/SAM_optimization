import os
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt

from model import get_model
from optimizer import get_optimizer

# sgd adam sam_sgd sam_adam esam_sgd esam_adam looksam_sgd looksam_adam dynamicsam_sgd dynamicsam_adam
METHOD = 'looksam_adam'
# resnet18 resnet34
MODEL = 'resnet34'
# 64 128 256
BATCH_SIZE = 256
NUM_EPOCHS = 100
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

LOG_DIR = f'log/{MODEL}/{BATCH_SIZE}'
RESULT_DIR = f'result/{MODEL}/{BATCH_SIZE}'
PICTURE_DIR = f'picture/{MODEL}/{BATCH_SIZE}'
WEIGHT_DIR = f'weight/{MODEL}/{BATCH_SIZE}'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(PICTURE_DIR):
    os.makedirs(PICTURE_DIR)
if not os.path.exists(WEIGHT_DIR):
    os.makedirs(WEIGHT_DIR)

LOG_PATH = f'{LOG_DIR}/training_{METHOD}.log'
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def calculate_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad(): 
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def calculate_loss(loader, model, criterion, device):
    total_loss = 0.0
    total_samples = 0

    model.eval()  
    with torch.no_grad(): 
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def add_gaussian_noise(inputs, mean=0.0, std=0.01):
    """给输入的图像添加高斯噪声"""
    # 生成与输入相同大小的噪声
    # [-1, 1] * std + mean
    noise = torch.randn_like(inputs) * std + mean
    print(f"---------- noise ----------")
    print(noise.size())
    print(noise)
    print(f"----------------------------")
    # 将噪声加入输入图像
    noisy_inputs = inputs + noise
    # 确保像素值在合理范围内（[0, 1]）
    # noisy_inputs = torch.clamp(noisy_inputs, 0.0, 1.0)
    return noisy_inputs

def main():
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model(model_name=MODEL, 
                      device=DEVICE)
    optimizer = get_optimizer(model=model, 
                              optimizer_name=METHOD, 
                              lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    train_loss_per_batch = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if 'sam' not in METHOD: # 不使用 SAM
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            elif 'esam' in METHOD: # 使用 ESAM
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                loss = closure()
                running_loss += loss.item()
                optimizer.first_step(zero_grad=True)

                closure()
                optimizer.second_step(zero_grad=True)
            elif 'looksam' in METHOD: # 使用 LookSAM
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                loss = closure()
                running_loss += loss.item()
                optimizer.step(t=i, samples=inputs, targets=labels, zero_grad=True)
            elif 'dynamicsam' in METHOD: # 使用 Dynamic SAM
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                loss = closure()
                running_loss += loss.item()
                optimizer.first_step(zero_grad=True, loss=loss)

                closure()
                optimizer.second_step(zero_grad=True)
            else: # 使用 SAM 
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    return loss

                loss = closure()
                running_loss += loss.item()
                optimizer.first_step(zero_grad=True)

                closure()
                optimizer.second_step(zero_grad=True)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = calculate_accuracy(train_loader, model, DEVICE)
        test_accuracy = calculate_accuracy(test_loader, model, DEVICE)
        avg_test_loss = calculate_loss(test_loader, model, criterion, DEVICE)  

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        logging.info(f"Average Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    logging.info('Finished Training')

    # 保存模型权重
    torch.save(model.state_dict(), f'{WEIGHT_DIR}/{MODEL}_{METHOD}.pth')
    # 保存训练结果
    logging.info('Saving training results')
    with open(f'{RESULT_DIR}/training_results_{METHOD}.txt', 'w') as f:
        f.write(f"train_losses: {train_losses}\n")
        f.write(f"test_losses: {test_losses}\n")
        f.write(f"train_accuracies: {train_accuracies}\n")
        f.write(f"test_accuracies: {test_accuracies}\n")
        # f.write(f"train_loss_per_batch: {train_loss_per_batch}\n")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linestyle='-', marker='o')
    plt.plot(test_losses, label='Test Loss', color='red', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green', linestyle='-', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange', linestyle='-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PICTURE_DIR}/training_results_{METHOD}.png')


if __name__ == '__main__':
    main()
