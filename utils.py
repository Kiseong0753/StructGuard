from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from advertorch.attacks import L2PGDAttack
from torchvision import datasets, transforms
from typing import *
import copy
import itertools
from itertools import cycle
from resnet_cifar100 import resnet18, resnet34, resnet50
import numpy as np
import random
from tqdm import tqdm
import glob
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class UTKDataset(Dataset):
    def __init__(self, root, age_grouping, transform):
        self.path = root
        file_list = glob.glob(self.path + "/*.jpg")
        self.data = []

        if age_grouping == 'TNN': # grouping from https://github.com/ArminBaz/UTK-Face/tree/master
            self.bins = np.array([0,10,15,20,25,30,40,50,60,120])
        elif age_grouping == 'MFD': # grouping from https://arxiv.org/abs/2106.04411
            self.bins = np.array([0,20,41,120])
        elif age_grouping == 'balanced': # balanced grouping:
            self.bins = np.array([0,20,30,45,120])
        elif age_grouping == 'groups': # teen, adult, etc.
            self.bins = np.array([0,4,13,20,31,46,61,120])
        elif age_grouping == 'tens':
            self.bins = np.array([0,11,21,31,41,51,61,71,81,91,101,111,120])
        else:
            raise NotImplementedError
        
        self.transform = transform
        
        for f in file_list:
            age = int(f.split('_')[0].split('/')[-1])
            class_name = np.where(age < self.bins)[0][0] - 1
            self.data.append([f, class_name])
            
    def __len__(self):
        
        return len(self.data)
        
    def __getitem__(self, idx):

        f, class_name = self.data[idx]
        img = Image.open(f)
        
        if self.transform:
            img = self.transform(img)

        #img = torch.from_numpy(img)
        #img = img.permute(2,0,1).float()
        
        return img, class_name

class FeatureExtractor:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Ahmed9275/Vit-Cifar100")

    def __call__(self, image):
        output = self.feature_extractor(list(image.unsqueeze(0).cpu()), return_tensors="pt")
        return output['pixel_values'][0]

class ViTModel(nn.Module):
    
    # Load pretrained ViT model
    def __init__(self):
        super(ViTModel, self).__init__()
        
        # Load ViT finetuned on Cifar100 https://huggingface.co/Ahmed9275/Vit-Cifar100
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Ahmed9275/Vit-Cifar100")
        self.encoder = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
        
    def forward(self, x):
        x = self.encoder(pixel_values=x)
        return x.logits

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self._len = len(inputs)

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


class NormalizeLayer(nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, device, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).to(device)
        self.sds = torch.tensor(sds).to(device)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds

def getDataLoaders(unlearn_k: int,
                   unlearn_label: int,
                   train_dataset,
                   test_dataset,
                   naive_unlearn_kwargs,
                   test_kwargs):
                   
    train_labels = torch.from_numpy(np.array(train_dataset.targets))

    #select unlearning data
    indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]

    copy_train_labels = train_labels.clone()
    copy_train_labels[indices_k_unlearn] = -10

    indices_other_data = (copy_train_labels != -10).nonzero(as_tuple=False)

    f_dataset = Subset(train_dataset, indices_k_unlearn.view(-1,))
    f_loader = torch.utils.data.DataLoader(f_dataset,**naive_unlearn_kwargs)

    r_dataset = Subset(train_dataset, indices_other_data.view(-1,))
    r_loader = torch.utils.data.DataLoader(r_dataset,**test_kwargs)

    t_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    print ('len(forget_dataset) : ', len(f_dataset), ' ',
           'len(residual_dataset) : ', len(r_dataset), ' ',
           'len(test_dataset) : ', len(test_dataset))
    
    return f_loader, r_loader, t_loader
    
def naive_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    CE = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = -CE(output, target)
        loss.backward()
        optimizer.step()
        
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    CE = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = CE(output, target)
        loss.backward()
        optimizer.step()
        
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    CE = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #output = output.to(target.device)

            test_loss += CE(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, 100. * correct / len(test_loader.dataset)


def test_ours(model, projector, device, test_loader):
    model.eval()
    projector.eval()
    test_loss = 0
    correct = 0
    CE = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, "feature")
            #output = output.to(target.device)
            output = projector(output)
            output = model.classification(output)
            test_loss += CE(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss, 100. * correct / len(test_loader.dataset)


def test_ours_utk(model, projection_mlp, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    CE = nn.CrossEntropyLoss()
    projection_mlp.eval()

    # extract the correct classification head
    if isinstance(model, torch.nn.DataParallel):
        classifier = model.module.fc
    else:
        classifier = model.fc

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, "feature")
            #output = output.to(target.device)
            output = projection_mlp(output)
            output = classifier(output)
            test_loss += CE(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, 100. * correct / len(test_loader.dataset)

def adv_attack(args, model, device, train_loader, adversary, unlearn_k, num_classes=100, num_adv_images = None, indices=None):
    model.eval()
    
    attacked_image_arr = []
    target_label_arr = []
    image_idx = []
    
    num_iters = args.num_adv_images
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        for i in tqdm(range(num_iters)):
            
            attack_label = torch.rand(data.shape[0]).to(device) * num_classes
            attack_label = attack_label.to(torch.long)
            attack_label = torch.where(attack_label == target, (torch.rand(data.shape[0]).long().to(device)*num_classes + num_classes) // 2, attack_label)

            adv_example = adversary.perturb(data, attack_label)

            inputs_numpy = adv_example.detach().cpu().numpy()
            labels_numpy = attack_label.cpu().numpy()

            for j in range(inputs_numpy.shape[0]):

                attacked_image_arr.append(inputs_numpy[j])
                target_label_arr.append(labels_numpy[j])                
            
    return attacked_image_arr, target_label_arr, np.unique(target_label_arr) 



def imagenet_adv_attack(args, model, device, train_loader, adversary, unlearn_k, num_classes=1000, num_adv_images=None, indices=None):
    model.eval()
    attacked_image_arr = []
    target_label_arr = []
    image_idx = []
    num_iters = args.num_adv_images
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        for i in tqdm(range(num_iters)):
            attack_label = (torch.rand(data.shape[0]).to(device) * (num_classes-1)).to(torch.long)
            
            is_same = (attack_label == target)
            if is_same.any():
                replacement = torch.randint(0, num_classes-1, size=(is_same.sum(),), device=device)
                target_for_same = target[is_same]
                replacement = torch.where(replacement >= target_for_same, replacement + 1, replacement)
                attack_label[is_same] = replacement
            
            attack_label = torch.clamp(attack_label, 0, num_classes-1)
            
            print(f"Attack labels min: {attack_label.min().item()}, max: {attack_label.max().item()}")
            
            adv_example = adversary.perturb(data, attack_label)
            
            inputs_numpy = adv_example.detach().cpu().numpy()
            labels_numpy = attack_label.cpu().numpy()
            
            for j in range(inputs_numpy.shape[0]):
                attacked_image_arr.append(inputs_numpy[j])
                target_label_arr.append(labels_numpy[j])
                
    return attacked_image_arr, target_label_arr, np.unique(target_label_arr)




def estimate_parameter_importance(trn_loader, model, device, num_samples, optimizer):
    importance = {n: torch.zeros(p.shape).to(device) for n, p in model.named_parameters()
                  if p.requires_grad}
    
    n_samples_batches = (num_samples // trn_loader.batch_size + 1) if num_samples > 0 \
        else (len(trn_loader.dataset) // trn_loader.batch_size)
    model.train()
    for images, targets in itertools.islice(trn_loader, n_samples_batches):
        outputs = model.forward(images.to(device))
        loss = torch.norm(outputs, p=2, dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.abs() * len(targets)
    n_samples = n_samples_batches * trn_loader.batch_size
    importance = {n: (p / n_samples) for n, p in importance.items()}
    return importance


def estimate_structure_aware_importance(loader, projector, origin_model, model, device, num_samples, semantic_anchors, constraint_type="Fisher", tau=1.0):
    model.train()
    importance = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total_samples = 0
    used_batches = 0
    batch_iter = iter(loader)

    while total_samples < num_samples:
        try:
            images, _ = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            images, _ = next(batch_iter)

        images = images.to(device)
        B = images.size(0)
        total_samples += B

        # Current embedding
        curr_feat = model(images, "feature")
        curr_feat = projector(curr_feat)
        curr_image_emb_norm = F.normalize(curr_feat, p=2, dim=1)
        curr_image_emb_norm = curr_image_emb_norm.float()

        # Frozen original embedding
        with torch.no_grad():
            orig_feat = origin_model(images, "feature").detach()
            orig_image_emb_norm = F.normalize(orig_feat, p=2, dim=1)
            orig_image_emb_norm = orig_image_emb_norm.float()

        # Compute similarity
        original_structure = orig_image_emb_norm @ semantic_anchors.T
        unlearned_structure = curr_image_emb_norm @ semantic_anchors.T

        orig_flat = original_structure.view(original_structure.size(0), -1)  # (B, C)
        curr_flat = unlearned_structure.view(unlearned_structure.size(0), -1)  # (B, C)
                    
        cos_sim = F.cosine_similarity(orig_flat, curr_flat, dim=1)  # (B,)
        loss = 1 - cos_sim.mean()
        
        model.zero_grad()
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.abs() * B

        used_batches += 1
    # Normalize by sample count
    importance = {k: v / total_samples for k, v in importance.items()}
    return importance



def estimate_structure_aware_importance_ResNet50(loader, dim_matcher, origin_model, model, device, num_samples, semantic_anchors, orig_projection, constraint_type="Fisher", tau=1.0):
    model.train()

    importance = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total_samples = 0
    used_batches = 0
    batch_iter = iter(loader)

    while total_samples < num_samples:
        try:
            images, _ = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            images, _ = next(batch_iter)

        images = images.to(device)
        B = images.size(0)
        total_samples += B

        # Current embedding
        curr_feat = model(images, "feature")
        curr_feat = dim_matcher(curr_feat)
        curr_image_emb_norm = F.normalize(curr_feat, p=2, dim=1)
        curr_image_emb_norm = curr_image_emb_norm.float()

        # Frozen original embedding
        with torch.no_grad():
            orig_feat = origin_model(images, "feature").detach()
            orig_feat = orig_feat @ orig_projection
            orig_image_emb_norm = F.normalize(orig_feat, p=2, dim=1)
            orig_image_emb_norm = orig_image_emb_norm.float()

        # Compute similarity
        original_structure = orig_image_emb_norm @ semantic_anchors.T
        unlearned_structure = curr_image_emb_norm @ semantic_anchors.T
        
        orig_flat = original_structure.view(original_structure.size(0), -1)  # (B, C)
        curr_flat = unlearned_structure.view(unlearned_structure.size(0), -1)  # (B, C)
                    
        cos_sim = F.cosine_similarity(orig_flat, curr_flat, dim=1)  # (B,)
        loss = 1 - cos_sim.mean()
        
        model.zero_grad()
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                importance[n] += p.grad.abs() * B
        used_batches += 1
    importance = {k: v / total_samples for k, v in importance.items()}
    return importance




def elastic_net_penalty(weight: torch.Tensor, phi: float = 0.99):
    """
    Elastic-net penalty:
      phi * ||W||_1  +  0.5*(1-phi)*||W||_F^2
    """
    l1 = torch.norm(weight, p=1)
    fro_sq = torch.norm(weight, p='fro')**2
    return phi * l1 + 0.5 * (1.0 - phi) * fro_sq

