import argparse
import copy
from itertools import cycle
import json
import os
import random
import sys

import clip
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from advertorch.attacks import L2PGDAttack
from torch.utils.data import Subset
from torchvision import datasets, transforms

from resnet_cifar100 import ProjectionMLP_ResNet50, ModelWrapper, resnet50
from utils import (
    JointDataset,
    NormalizeLayer,
    adv_attack,
    elastic_net_penalty,
    estimate_parameter_importance,
    estimate_structure_aware_importance_ResNet50,
    naive_train,
    test,
    test_ours,
)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--pgd_eps', type=float, default=2.0, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--base_pgd_eps', type=float, default=2.0, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--pgd_alpha', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--pgd_iter', type=int, default=100, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    
    parser.add_argument('--unlearn_label', type=int, default=9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--unlearn_k', type=int, default=10, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--unlearn_lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--projection_lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--baseline_lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--num_adv_images', type=int, default=None, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--reg_lamb', type=float, default=10.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--log_name', type=str, default="CVPR26")
    parser.add_argument('--constraint_type', type=str, default="KL") # KL, JS, WA, MMD
    parser.add_argument('--weight_level', type=str, default="Fisher") # Fisher, MAS

    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = args.device
    if use_cuda:
        torch.cuda.set_device(device)

    log_dir = os.path.join('./logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger.add(log_dir + '/' + args.log_name +"_{time}.log")
    logger.add(sys.stdout, colorize=True, format="{message}")
    
    args_str = ', '.join(f'{k}={v}' for k, v in vars(args).items())
    logger.info(f'Parsed arguments: {args_str}')
   
    eps = args.pgd_eps
    iters = args.pgd_iter
    alpha = args.pgd_alpha
    
    k_arr = [16, 64, 128, 256]

    D_r_acc = []
    D_f_acc = []
    D_test_acc = []
    
    case1_D_r = []
    case2_D_r = []
    case3_D_r = []
    case4_D_r = []

    case1_D_f = []
    case2_D_f = []
    case3_D_f = []
    case4_D_f = []

    case1_D_test = []
    case2_D_test = []
    case3_D_test = []
    case4_D_test = []


    train_kwargs = {'batch_size': 128}
    test_kwargs = {'batch_size': 128}

    naiive_unlearn_kwargs = {'batch_size': 128}
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])

    dataset1 = datasets.CIFAR100('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR100('../data', train=False,
                       transform=transform)
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    for unlearn_k in k_arr:
        random_seed = 0
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)
        save_dir = 'CIFAR100_unlearned_model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        unlearn_label = args.unlearn_label

        train_labels = dataset1.targets
        
        train_labels = torch.from_numpy(np.array(train_labels))

        indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
        logger.info(f'indices_k_unlearn : {indices_k_unlearn}')

        copy_train_labels = train_labels.clone()
        copy_train_labels[indices_k_unlearn] = -10

        indices_other_data = (copy_train_labels != -10).nonzero(as_tuple=False)


        unlearn_dataset = Subset(dataset1, indices_k_unlearn.view(-1,))
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset,**naiive_unlearn_kwargs)

        other_dataset = Subset(dataset1, indices_other_data.view(-1,))
        other_loader = torch.utils.data.DataLoader(other_dataset,**test_kwargs)
        
        cifar_test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        
        logger.info(f'len(unlearn_dataset) : {len(unlearn_dataset)} || len(other_dataset) : {len(other_dataset)}')

        model = resnet50().to(device)
        model.load_state_dict(torch.load('pretrained_models/cifar100_pretrained_models/resnet50.pt', map_location=device))
        normalize_layer = NormalizeLayer(device, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        model = torch.nn.Sequential(normalize_layer, model)
        
        optimizer = optim.SGD(model.parameters(), lr=args.baseline_lr, momentum=0.9, weight_decay=1e-4)
        
        model.eval()

        other_loss, other_acc = test(model, device, other_loader)
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)
        
        str_list = '\n Before | D_test - D_forget acc : ' + str(other_acc) +  ', D_forget acc : ' + str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        
        D_test_acc.append(test_acc)
        D_r_acc.append(other_acc)
        D_f_acc.append(unlearn_acc)
        
        logger.info(str_list)

################################################################################################################################################################
        logger.info('Baseline 1 (Neggrad): finetuning with D_forget (maximizing CE loss)')
        unlearn_acc = 100
        max_iter = 100
        j = 0

        while unlearn_acc != 0:
            naive_train(args, model, device, unlearn_loader, optimizer, 0)
            model.eval()
            unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
            j += 1
            if max_iter < j:
                break
                
        model.eval()
        other_loss, other_acc = test(model, device, other_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)

        str_list = '\n After | D_test - D_retention acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc) +  ', D_test acc : ' +  str(test_acc)
        logger.info(str_list)
        
        model_path = os.path.join(save_dir, f'CIFAR100_{unlearn_k}__Naive_unlearned_model.pt')   
        torch.save(model.state_dict(), model_path)
        
        case1_D_test.append(test_acc)
        case1_D_r.append(other_acc)
        case1_D_f.append(unlearn_acc)
        
#################################################################################################################################################################
        logger.info('\n Baseline 2 (Adv): using adversarial examples only')
        model = resnet50().to(device)
        model.load_state_dict(torch.load('pretrained_models/cifar100_pretrained_models/resnet50.pt'))
        normalize_layer = NormalizeLayer(device, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        model = torch.nn.Sequential(normalize_layer, model)

        optimizer = optim.SGD(model.parameters(), lr=args.baseline_lr, momentum=0.9, weight_decay=1e-4)
        
        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        unlearn_acc = 100
        alpha = 0.0
        model.eval()
        adversary = L2PGDAttack(model, eps=args.base_pgd_eps, eps_iter=args.pgd_alpha, nb_iter=args.pgd_iter,
                                rand_init=True, targeted=True)
        adv_images, target_labels, class_id_for_adv_descriptions = adv_attack(args, model, device, unlearn_loader, adversary, unlearn_k, num_classes = args.num_class)
        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0
        
        unlearn_loader_cycle = cycle(unlearn_loader)
        CE = nn.CrossEntropyLoss()
        
        while unlearn_acc != 0:
            model.train()

            for i , data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
                model.train()
                (adv_data, adv_target), (data, target) = data
                optimizer.zero_grad()
                output_adv = model(adv_data.to(device))
                output = model(data.to(device))

                loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss = loss_unlearn + loss_adv

                loss.backward()
                optimizer.step()
                
                model.eval()
                unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
                
                if unlearn_acc == 0:
                    logger.info(f'unlearn_acc == 0, Break at j = {j}, i = {i}')
                    break
            j += 1
            if max_iter < j:
                break

        model.eval()
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        other_loss, other_acc = test(model, device, other_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)
        str_list = '\n After | D_test - D_retention acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        logger.info(str_list)

        model_path = os.path.join(save_dir, f'CIFAR100_{unlearn_k}_{args.base_pgd_eps}_Adv_unlearned_model.pt')   
        torch.save(model.state_dict(), model_path)
        
        case2_D_test.append(test_acc)
        case2_D_r.append(other_acc)
        case2_D_f.append(unlearn_acc)
#################################################################################################################################################################
        logger.info('\n Baseline 3 (L2UL): using both adversarial examples and weight importance')

        model = resnet50().to(device)
        model.load_state_dict(torch.load('pretrained_models/cifar100_pretrained_models/resnet50.pt'))
        normalize_layer = NormalizeLayer(device, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        model = torch.nn.Sequential(normalize_layer, model)

        optimizer = optim.SGD(model.parameters(), lr=args.baseline_lr, momentum=0.9, weight_decay=1e-4)
        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

        model_for_importance = copy.deepcopy(model)
        num_samples = len(unlearn_dataset)
        importance = estimate_parameter_importance(unlearn_loader, model_for_importance, device, num_samples, optimizer)

        for keys in importance.keys():
            importance[keys] = (importance[keys] - importance[keys].min()) / (importance[keys].max() - importance[keys].min())
            importance[keys] = (1 - importance[keys])
                    
        CE = nn.CrossEntropyLoss()

        unlearn_acc = 100
        adv_dataset = JointDataset(adv_images, target_labels)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0
        unlearn_loader_cycle = cycle(unlearn_loader)
        
        while unlearn_acc != 0:
            for i , data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
                model.train()
                (adv_data, adv_target), (data, target) = data
                optimizer.zero_grad()

                output_adv = model(adv_data.to(device))
                output = model(data.to(device))

                loss_unlearn = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                loss_adv = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))

                loss_reg = 0

                for n, p in model.named_parameters():
                    if n in importance.keys():
                        loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2

                loss = loss_unlearn + loss_adv + loss_reg * args.reg_lamb

                loss.backward()
                optimizer.step()
                model.eval()
                unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
                
                if unlearn_acc == 0:
                    logger.info(f'unlearn_acc == 0, Break at j = {j}, i = {i}')
                    break
            j += 1
            if max_iter < j:
                break
        
        model.eval()
        unlearn_loss, unlearn_acc = test(model, device, unlearn_loader)
        other_loss, other_acc = test(model, device, other_loader)
        test_loss, test_acc = test(model, device, cifar_test_loader)

        str_list = '\n After | D_test - D_retention acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        logger.info(str_list)
        
        model_path = os.path.join(save_dir, f'CIFAR100_{unlearn_k}_{args.base_pgd_eps}_Adv_Weight_unlearned_model.pt')   
        torch.save(model.state_dict(), model_path)
        
        case3_D_test.append(test_acc)
        case3_D_r.append(other_acc)
        case3_D_f.append(unlearn_acc)
##############################################################################################################################################################################        
        logger.info('\n Baseline 4 (Ours): StructGuard')

        adversary = L2PGDAttack(model, eps=args.pgd_eps, eps_iter=args.pgd_alpha, nb_iter=args.pgd_iter,
                                rand_init=True, targeted=True)

        adv_images, target_labels, class_id_for_adv_descriptions = adv_attack(args, model, device, unlearn_loader, adversary, unlearn_k, num_classes = args.num_class)
        
        text_encoder, preprocess = clip.load("ViT-B/32", device=device)
        with open("CIFAR100_Description.json", "r") as f:
            cifar100_attributes = json.load(f)

        class_sentences = { 
            class_name: "; ".join(attributes) 
            for class_name, attributes in cifar100_attributes.items() 
        }
        
        model = resnet50().to(device)
        model.load_state_dict(torch.load('pretrained_models/cifar100_pretrained_models/resnet50.pt', map_location=device))
        normalize_layer = NormalizeLayer(device, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        model = ModelWrapper(normalize_layer, model).to(device)

        projection_mlp = ProjectionMLP_ResNet50(input_dim=2048, hidden_dim=512, output_dim=100).to(device)
        optimizer = optim.SGD([
            {'params': model.parameters(), 'lr': args.unlearn_lr},
            {'params': projection_mlp.parameters(), 'lr': args.projection_lr}
        ], lr=args.unlearn_lr)

        original_model = copy.deepcopy(model).eval().to(device)
        for p in original_model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            texts = clip.tokenize(list(class_sentences.values()), truncate=True).to(device)
            semantic_anchors = text_encoder.encode_text(texts)
            semantic_anchors = semantic_anchors / semantic_anchors.norm(dim=-1, keepdim=True)
            semantic_anchors = semantic_anchors.float()
            orig_projection = torch.randn(2048, semantic_anchors.size(1), device=device)
            orig_projection = F.normalize(orig_projection, dim=0)
            
        origin_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        model_for_importance = copy.deepcopy(model)
        max_iter = 100

        unlearn_acc = 100
        adv_dataset = JointDataset(adv_images, target_labels)
        num_samples = len(other_dataset)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, **train_kwargs)

        j = 0
        
        unlearn_loader_cycle = cycle(unlearn_loader)
        CE = nn.CrossEntropyLoss()
        already_computed_importance = False

        while unlearn_acc != 0:
            model.train()

            for i , data in enumerate(zip(adv_loader, unlearn_loader_cycle)):
                model.train()
                (adv_data, adv_target), (data, target) = data
                optimizer.zero_grad()
                output = model(data.to(device), "classification")
                projector_input = model(adv_data.to(device), "feature")
                image_embedding = projection_mlp(projector_input)

                output_adv = model.classification(image_embedding)
                Deletion_loss = -CE(output, target.to(device)) * (data.shape[0] / (adv_data.shape[0] + data.shape[0]))
                Retention_loss = CE(output_adv, adv_target.to(device)) * (adv_data.shape[0] / (adv_data.shape[0] + data.shape[0]))   
                
                with torch.no_grad():
                    orig_feat = original_model(adv_data.to(device), "feature")
                    orig_feat = orig_feat @ orig_projection
                    orig_image_emb_norm = F.normalize(orig_feat, p=2, dim=1)
                    orig_image_emb_norm = orig_image_emb_norm.float()

                image_embedding = projection_mlp.mlp[:3](projector_input)
                curr_image_emb_norm = F.normalize(image_embedding, p=2, dim=1).float()
                curr_image_emb_norm = curr_image_emb_norm.float()
                original_structure = orig_image_emb_norm @ semantic_anchors.T
                unlearned_structure = curr_image_emb_norm @ semantic_anchors.T
                tau = getattr(args, "tau", 1.0)  
                    
                if args.constraint_type == "sim":
                    orig_flat = original_structure.view(original_structure.size(0), -1)  # (B, C)
                    curr_flat = unlearned_structure.view(unlearned_structure.size(0), -1)  # (B, C)
                    cos_sim = F.cosine_similarity(orig_flat, curr_flat, dim=1)  # (B,)
                    structure_alignment_loss = 1 - cos_sim.mean()
                else:
                    raise ValueError(f"Unknown constraint_type: {args.constraint_type}")

                if i == 1 and not already_computed_importance: 
                        importance = estimate_structure_aware_importance_ResNet50(adv_loader, copy.deepcopy(projection_mlp.mlp[:3]), model_for_importance, model, device, num_samples, semantic_anchors, orig_projection, args.weight_level)
                        optimizer.zero_grad(set_to_none=True)
                        already_computed_importance = True
                if i > 1:
                        structure_regularization_loss = 0
                        for n, p in model.named_parameters():
                            if n in importance.keys():
                                structure_regularization_loss += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2
                else:
                        structure_regularization_loss = 0
                        
                Wl = model.fc.weight
                fc_reg_loss = elastic_net_penalty(Wl, 0.99)

                total_loss = Deletion_loss + Retention_loss + structure_alignment_loss + args.reg_lamb*structure_regularization_loss + 0.001*fc_reg_loss
                total_loss.backward()
                optimizer.step()
                model.eval()
            
                if j > 0:
                        unlearn_loss, unlearn_acc = test_ours(model, projection_mlp, device, unlearn_loader)
                        if unlearn_acc == 0:
                            logger.info(f'unlearn_acc == 0, Break at j = {j}, i = {i}')
                            break

            j += 1
            if max_iter < j:
                break

        model.eval()

        unlearn_loss, unlearn_acc = test_ours(model, projection_mlp, device, unlearn_loader)
        other_loss, other_acc = test_ours(model, projection_mlp, device, other_loader)
        test_loss, test_acc = test_ours(model, projection_mlp, device, cifar_test_loader)
        str_list = '\n After | D_test - D_retention acc : ' + str(other_acc) +  ', D_forget acc : ' +  str(unlearn_acc)+  ', D_test acc : ' +  str(test_acc)
        logger.info(str_list)
            
        model_path = os.path.join(save_dir, f'CIFAR100_{unlearn_k}_{args.pgd_eps}_Ours_unlearned_model.pt')   
        torch.save(model.state_dict(), model_path)
        
        case4_D_test.append(test_acc)
        case4_D_r.append(other_acc)
        case4_D_f.append(unlearn_acc)         


if __name__ == '__main__':
    main()
