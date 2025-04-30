# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import shift, rotate, zoom

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]

def augment_images(images, labels, num_augmentations=2):
    """
    Augment images by applying random transformations.
    images: [num_samples, 28*28]
    labels: [num_samples]
    """
    augmented_images = []
    augmented_labels = []
    
    for img, label in zip(images, labels):
        img_2d = img.reshape(28, 28)

        augmented_images.append(img)
        augmented_labels.append(label)

        for _ in range(num_augmentations):
            shift_x = np.random.uniform(-2, 2)
            shift_y = np.random.uniform(-2, 2)
            shifted_img = shift(img_2d, [shift_y, shift_x], mode='nearest')

            angle = np.random.uniform(-10, 10)
            rotated_img = rotate(shifted_img, angle, reshape=False, mode='nearest')

            scale = np.random.uniform(0.9, 1.1)
            scaled_img = zoom(rotated_img, scale, mode='nearest')

            if scaled_img.shape != (28, 28):
                scaled_img = zoom(scaled_img, (28 / scaled_img.shape[0], 28 / scaled_img.shape[1]), mode='nearest')

            augmented_images.append(scaled_img.flatten())
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)

train_imgs, train_labs = augment_images(train_imgs[10000:], train_labs[10000:], num_augmentations=2)

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 插入 reshape 和 Model_CNN
train_imgs = train_imgs.reshape(-1, 1, 28, 28)  # [num_samples, 1, 28, 28]
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

linear_model = nn.models.Model_CNN(size_list=[1, 16, 32, 128, 10], lambda_list=[1e-4, 1e-4, 1e-4, 1e-4], kernel_size=3)

optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

conv_layer = linear_model.layers[0]  # Display the first convolutional layer.
kernels = conv_layer.W  # [out_channels, in_channels, k_H, k_W], here [16, 1, 3, 3]

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
axes = axes.flatten()
for i in range(min(16, kernels.shape[0])):  # Show only the first 16 kernels.
    kernel = kernels[i, 0]  # [3, 3]
    axes[i].imshow(kernel, cmap='gray')
    axes[i].set_title(f'Kernel {i}')
    axes[i].axis('off')
plt.tight_layout()
plt.show()
plt.savefig('kernels.png')
