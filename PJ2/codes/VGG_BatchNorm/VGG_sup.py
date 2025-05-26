import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display
import math

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader


device_id = [0, 1, 2, 3]
num_workers = 4
batch_size = 128
epochs_n = 40
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

figures_path = ''
models_path = ''

device = 'cuda'

train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers)
val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=num_workers)
batches_per_epoch = len(train_loader)  # 应为 ceil(50000 / 128) = 391

def get_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0
    losses_list = []

    batches_n = len(train_loader)
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()
        loss_list = []
        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        losses_list.append(loss_list)
       
        learning_curve[epoch] = np.mean(loss_list)
    
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader)
        val_accuracy = get_accuracy(model, val_loader)
        val_accuracy_curve[epoch] = val_accuracy
     
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{epochs_n}, Loss: {learning_curve[epoch]:.4f}, "
              f"Train Acc: {train_accuracy_curve[epoch]:.2f}%, Val Acc: {val_accuracy_curve[epoch]:.2f}%")
        
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 3, figsize=(15, 3))
        axes[0].plot(learning_curve)
        axes[0].set_title("Training Loss")
        axes[1].plot(train_accuracy_curve)
        axes[1].set_title("Training Accuracy")
        axes[2].plot(val_accuracy_curve)
        axes[2].set_title("Validation Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'training_curve_bn_epoch_{epoch+1}.png'))
        plt.close()

    return losses_list, learning_curve, train_accuracy_curve, val_accuracy_curve

vgg_a_losses = {}
for lr in learning_rates:
    loss_file = os.path.join(models_path, f'vgg_a_loss_lr_{lr}.txt')
    if os.path.exists(loss_file):
        flat_losses = np.loadtxt(loss_file, dtype=float)
        print(f"Loaded {len(flat_losses)} losses for lr={lr} from {loss_file}")
        
        total_batches = len(flat_losses)
        actual_epochs = math.ceil(total_batches / batches_per_epoch)
        print(f"Total batches: {total_batches}, Batches per epoch: {batches_per_epoch}, Actual epochs: {actual_epochs}")
        
        if total_batches >= epochs_n * batches_per_epoch:
            flat_losses = flat_losses[:epochs_n * batches_per_epoch]
        else:
            flat_losses = np.pad(flat_losses, (0, epochs_n * batches_per_epoch - total_batches), 
                               mode='constant', constant_values=flat_losses[-1])
        
        losses = np.array(flat_losses).reshape(epochs_n, batches_per_epoch).tolist()
        vgg_a_losses[lr] = losses
        print(f"Reshaped VGG_A losses for lr={lr} to shape: {np.array(losses).shape}")
    else:
        print(f"Warning: Loss file for lr={lr} not found at {loss_file}")

# VGG_A_BatchNorm
set_random_seeds(seed_value=2020, device=device)
vgg_a_bn_losses = {}
for lr in learning_rates:
    print(f"Training VGG_A_BatchNorm with lr={lr}")
    model = VGG_A_BatchNorm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_model_path = os.path.join(models_path, f'vgg_a_bn_lr_{lr}.pth')
    losses, _, _, _ = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epochs_n, best_model_path=best_model_path)
    vgg_a_bn_losses[lr] = losses
    np.savetxt(os.path.join(models_path, f'vgg_a_bn_loss_lr_{lr}.txt'), np.array(losses).flatten(), fmt='%f')
    print(f"Saved VGG_A_BatchNorm losses for lr={lr} to {models_path}/vgg_a_bn_loss_lr_{lr}.txt")

# max_curve, min_curve
min_curve_vgg_a = []
max_curve_vgg_a = []
min_curve_vgg_a_bn = []
max_curve_vgg_a_bn = []

for epoch in range(epochs_n):
    epoch_losses_vgg_a = []
    epoch_losses_vgg_a_bn = []
    for lr in learning_rates:
        if lr in vgg_a_losses:
            epoch_losses_vgg_a.extend(vgg_a_losses[lr][epoch])
        epoch_losses_vgg_a_bn.extend(vgg_a_bn_losses[lr][epoch])
    min_curve_vgg_a.append(np.min(epoch_losses_vgg_a) if epoch_losses_vgg_a else np.nan)
    max_curve_vgg_a.append(np.max(epoch_losses_vgg_a) if epoch_losses_vgg_a else np.nan)
    min_curve_vgg_a_bn.append(np.min(epoch_losses_vgg_a_bn))
    max_curve_vgg_a_bn.append(np.max(epoch_losses_vgg_a_bn))

def plot_loss_landscape():
    plt.figure(figsize=(10, 6))
    epochs = range(1, epochs_n + 1)
    
    plt.plot(epochs, max_curve_vgg_a, 'b-', label='VGG_A Max Loss')
    plt.plot(epochs, min_curve_vgg_a, 'b--', label='VGG_A Min Loss')
    plt.fill_between(epochs, min_curve_vgg_a, max_curve_vgg_a, color='blue', alpha=0.2, label='VGG_A Loss Range')
    
    plt.plot(epochs, max_curve_vgg_a_bn, 'r-', label='VGG_A_BN Max Loss')
    plt.plot(epochs, min_curve_vgg_a_bn, 'r--', label='VGG_A_BN Min Loss')
    plt.fill_between(epochs, min_curve_vgg_a_bn, max_curve_vgg_a_bn, color='red', alpha=0.2, label='VGG_A_BN Loss Range')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Landscape: VGG_A vs VGG_A_BatchNorm')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))
    plt.close()
    print(f"Loss landscape saved to {figures_path}/loss_landscape.png")

plot_loss_landscape()