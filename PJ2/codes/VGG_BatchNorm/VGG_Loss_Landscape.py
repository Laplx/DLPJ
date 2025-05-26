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

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
figures_path = ''
models_path = ''


# # Make sure you are using the right device.
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
# print(device)
# print(torch.cuda.get_device_name(3))

print(torch.cuda.is_available())
device = 'cuda'

# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    print(f"Sample batch shape: {X.shape}, Labels: {y.shape}")
    break


# This function is used to calculate the accuracy of model classification
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

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            loss.backward()
            optimizer.step()

            grad = model.classifier[4].weight.grad.clone()
            loss_list.append(loss.item())


        losses_list.append(loss_list)
        grads.append(grad)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
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
        plt.savefig(os.path.join(figures_path, f'training_curve_epoch_{epoch+1}.png'))
        plt.close()

    return losses_list, grads


# Train your model
# feel free to modify
epo = 20
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

set_random_seeds(seed_value=2020, device=device)

# VGG_A
vgg_a_losses = {}
for lr in learning_rates:
    print(f"Training VGG_A with lr={lr}")
    model = VGG_A().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_model_path = os.path.join(models_path, f'vgg_a_lr_{lr}.pth')
    losses, _ = train(model, optimizer, criterion, train_loader, val_loader, best_model_path=best_model_path)
    vgg_a_losses[lr] = losses
    np.savetxt(os.path.join(models_path, f'vgg_a_loss_lr_{lr}.txt'), np.array(losses).flatten(), fmt='%f')

# VGG_A_BatchNorm
vgg_a_bn_losses = {}
for lr in learning_rates:
    print(f"Training VGG_A_BatchNorm with lr={lr}")
    model = VGG_A_BatchNorm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_model_path = os.path.join(models_path, f'vgg_a_bn_lr_{lr}.pth')
    losses, _ = train(model, optimizer, criterion, train_loader, val_loader, best_model_path=best_model_path)
    vgg_a_bn_losses[lr] = losses
    np.savetxt(os.path.join(models_path, f'vgg_a_bn_loss_lr_{lr}.txt'), np.array(losses).flatten(), fmt='%f')


# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve_vgg_a = []
max_curve_vgg_a = []
min_curve_vgg_a_bn = []
max_curve_vgg_a_bn = []

for epoch in range(epochs_n):
    epoch_losses_vgg_a = []
    epoch_losses_vgg_a_bn = []
    for lr in learning_rates:
        epoch_losses_vgg_a.extend(vgg_a_losses[lr][epoch])
        epoch_losses_vgg_a_bn.extend(vgg_a_bn_losses[lr][epoch])
    min_curve_vgg_a.append(np.min(epoch_losses_vgg_a))
    max_curve_vgg_a.append(np.max(epoch_losses_vgg_a))
    min_curve_vgg_a_bn.append(np.min(epoch_losses_vgg_a_bn))
    max_curve_vgg_a_bn.append(np.max(epoch_losses_vgg_a_bn))


# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    plt.figure(figsize=(10, 6))
    epochs = range(1, epochs_n + 1)
    
    # VGG_A
    plt.plot(epochs, max_curve_vgg_a, 'b-', label='VGG_A Max Loss')
    plt.plot(epochs, min_curve_vgg_a, 'b--', label='VGG_A Min Loss')
    plt.fill_between(epochs, min_curve_vgg_a, max_curve_vgg_a, color='blue', alpha=0.2, label='VGG_A Loss Range')
    
    # VGG_A_BatchNorm
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
    return

plot_loss_landscape()