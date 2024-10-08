import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
!pip install -q foolbox
import foolbox as fb
!pip install -q git+https://github.com/RobustBench/robustbench.git
from autoattack import AutoAttack
from torchvision.models.resnet import Bottleneck
import multiprocessing
from tqdm import tqdm
import random
import time

import matplotlib.pyplot as plt
import torchvision.transforms.functional as FF


#!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #self.mode = mode
        self.backbone = models.resnet152(pretrained=True)

        self.backbone.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num, 100)

    #def set_mode(self, mode):
    #  self.mode = mode
    
    def forward(self, x):
        #if self.mode == "Origin":
            x = self.backbone(x)
            return x

class RandomJitterXY:
    def __init__(self, max_translation=3):
        self.max_translation = max_translation
    def __call__(self, img):
        translate_x = random.uniform(-self.max_translation, self.max_translation)
        translate_y = random.uniform(-self.max_translation, self.max_translation)
        return transforms.functional.affine(img, angle=0, translate=(translate_x, translate_y), scale=1, shear=0)

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.2):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        a = torch.randn(tensor.size())
        #print(a)
        noise = torch.randn(tensor.size()) * self.std + self.mean
        #print(tensor.size())
        noise_img = tensor + noise
        return torch.clamp(noise_img, 0, 1)  

class MultiResolutionTransform:
    def __init__(self):
        self.resolutions = [32, 16, 8, 4]
        self.transforms = [transforms.Compose([
            transforms.Resize((res, res), interpolation=Image.BICUBIC),
            RandomJitterXY(max_translation=3),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.2),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.5), 
            transforms.RandomApply([transforms.RandomGrayscale(p=0.2)], p=0.5),    
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            AddGaussianNoise(mean=0.0, std=0.2)
        ]) for res in self.resolutions]
    def __call__(self, image):
        #print(image.shape)
        images = [transform(image) for transform in self.transforms]
        multi_res_image = torch.cat(images, dim=0)
        #print(multi_res_image.shape)
        return multi_res_image

class MultiResolutionTestTransform:
    def __init__(self):
        self.resolutions = [32, 16, 8, 4]
        self.transforms = [transforms.Compose([
            transforms.Resize((res, res), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Resize((224, 224))
        ]) for res in self.resolutions]
    def __call__(self, image):
        images = [transform(image) for transform in self.transforms]
        multi_res_image = torch.cat(images, dim=0)
        return multi_res_image

criterion = nn.CrossEntropyLoss()

def train(model, max, train, optim, cri, sche):
  for epoch in range(max):
          running_loss = 0.0
          for image, label in tqdm(train, desc=f"Epoch [{epoch+1}] : "):
              image, label = image.to(device), label.to(device)
              optim.zero_grad()
              out = model(image)
              loss = cri(out, label)
              loss.backward()
              optim.step()
              running_loss += loss.item()
          sche.step()
          print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def train_prob(model, train, optim, cri):
  for epoch in range(2):
    for image, label in tqdm(train, desc=f"Epoch"):
      image, label = image.to(device), label.to(device)
      optim.zero_grad()
      out, linear_output = model(image)
      linear_losses = 0
      for output in linear_output:
        linear_loss = cri(output, label)
        linear_losses += linear_loss
      linear_losses.backward()
      optim.step()

def test(model, test):
  with torch.no_grad():
    total = 0
    correct = 0
    for image, label in tqdm(test_loader, desc=f""):
        image, label = image.to(device), label.to(device)
        out = model(image)
        _, pred = torch.max(out.data, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()
    print(f"Total : {total}, Correct : {correct}")

def test_prob(model, test):
  total = [0] * 3
  correct = [0] * 3
  for image, label in tqdm(test, desc=f""):
    image, label = image.to(device), label.to(device)
    #print(f"Image : image.shape")
    _, out = model(image)
    for i in range(len(out)):
      output = out[i]
      _, pred = torch.max(output.data, 1)
      total[i] += label.size(0)
      correct[i] += (pred==label).sum().item()

def crossmax(logit, b, n, c):
  logit_ = logit - torch.max(logit, dim=2, keepdim=True)[0]
  logit_ = logit_ - torch.max(logit_, dim=1, keepdim=True)[0]
  logit_ = torch.median(logit_, dim=1)[0]
  return logit_

train_dataset = Datasets.CIFAR100(root='./data', train=True, download=True, transform=MultiResolutionTransform())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=8)

test_dataset = Datasets.CIFAR100(root='./data', train=False, download=True, transform=MultiResolutionTestTransform())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=8)

def adv(image, label, model):
        x = image.clone().detach().requires_grad_(True)
        a = model(x)
        if a.dim() == 1:
            a = a.unsqueeze(0)
        loss = criterion(a, label)
        loss.backward()
        x_adv = x + 0.1 * x.grad.sign()
        return x_adv

"""
def visualize_augmentation_stepwise(image, transform):
    #plt.figure(figsize=(15, 5))
    
    
    plt.subplot(1, 6, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    res = [32, 16, 8, 4]
    tr = [transforms.Compose([
        transforms.Resize((res_, res_)),
        RandomJitterXY(max_translation=3),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std=0.2),
        transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.5), 
        transforms.RandomApply([transforms.RandomGrayscale(p=0.2)], p=0.5),  
        transforms.Resize((32, 32)),
        AddGaussianNoise(mean=0, std=0.2)
    ]) for res_ in res]
    transform_img = [tr_(image) for tr_ in tr]
    
    transform_img = torch.cat(transform_img, dim=0)
    print(transform_img.shape)
    a = FF.to_pil_image(transform_img[0:3, :, :])
    b = FF.to_pil_image(transform_img[1, :, :])
    c = FF.to_pil_image(transform_img[2, :, :])
    #d = FF.to_pil_image(transform_img[10, :, :])
    plt.subplot(1,6,2)
    plt.title('dd')
    plt.imshow(a)
    plt.axis('off')

    plt.subplot(1,6,3)
    plt.title('dd')
    plt.imshow(b)
    plt.axis('off')

    plt.subplot(1,6,4)
    plt.title('dd')
    plt.imshow(c)
    plt.axis('off')

    #plt.subplot(1,6,5)
    #plt.title('dd')
    #plt.imshow(d)
    #plt.axis('off')
    
    #jitter_transform = RandomJitterXY(max_translation=3)
    #jittered_image = jitter_transform(transform_img)
    
    #plt.subplot(1, 4, 2)
    #plt.title('Random Jitter (PIL)')
    #plt.imshow(jittered_image)
    #plt.axis('off')
    
    
    #tensor_image = FF.to_tensor(jittered_image)  
    #noise_transform = AddGaussianNoise(mean=0.0, std=0.2)
    #noised_image = noise_transform(tensor_image)
    
    
    #print(f"Tensor values before Gaussian Noise -> Min: {tensor_image.min()}, Max: {tensor_image.max()}")
    #print(f"Tensor values after Gaussian Noise -> Min: {noised_image.min()}, Max: {noised_image.max()}")
    
    
    #plt.subplot(1, 4, 3)
    #plt.title('Gaussian Noise (Tensor)')
    #plt.imshow(FF.to_pil_image(noised_image))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
"""


def main():
    #sample_image, _ = train_dataset[1700]  # CIFAR-100에서 샘플
    #visualize_augmentation_stepwise(sample_image, MultiResolutionTransform())

    max_epoch = 1
    criterion = nn.CrossEntropyLoss()
    model = ResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr= 3.3e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    train(model, max_epoch, train_loader, optimizer, criterion, scheduler)
    torch.save(model, 'multi_152_224_c100.pth')


    model.eval()
    test(model, test_loader)
    """
    x_test = []
    y_test = []
    for i, (image, label) in enumerate(test_loader):
      image, label = image.to(device), label.to(device)
      x_test.append(image)
      y_test.append(label)
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test_tensor, y_test_tensor)
    """

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()