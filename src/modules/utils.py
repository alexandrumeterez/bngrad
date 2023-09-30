import torch
import numpy as np
from scipy.stats import gmean
import tqdm
import math
import sys

def generate_matrix_close_to_isometry(d):
    X = torch.rand(d, d)
    eps = torch.rand(d)
    eps /= eps.sum()
    eps -= 1/d

    eigs = (1 + eps).sqrt()
    U, S, Vt = torch.linalg.svd(X)
    return U @ eigs.diag() @ Vt

def generate_matrix_far_from_isometry(d, eps):
    X = torch.rand(d, d)
    eigs = torch.from_numpy(np.asarray([d - (d-1) * eps] + [eps] * (d-1))).float().sqrt()
    U, S, Vt = torch.linalg.svd(X)
    return U @ eigs.diag() @ Vt

def cosine(x, y):
    return torch.dot(x, y) / (x.norm().abs() * y.norm().abs())

def cosine_similarity(x, y):
    return 1 - cosine(x, y).abs()

def isometry_gap(X):
    G = X @ X.t()
    G = G.detach().cpu()
    eigs = torch.linalg.eigvalsh(G)
    return -torch.log(eigs).mean() + torch.log(torch.mean(eigs))

def ortho_gap(X):
    n, d = X.shape
    I_n = torch.eye(n).to(X.device)
    Y1 = (X@X.T) / (X.norm('fro')**2)
    Y2 = I_n / (I_n.norm('fro')**2)
    return (Y1 - Y2).norm('fro')

def isometry_gap2(X):
    G = X @ X.t()
    G = G.detach().cpu()
    eigs = torch.linalg.eigvalsh(G).numpy()
    return -np.log(gmean(eigs) / eigs.mean())


def get_measurements(model, inputs, labels, criterion, epoch, device):
    # Do one forward pass and one backward pass without updating anything
    model.train()
    inputs, labels = inputs.to(device), labels.to(device)
    model.set_save_hidden(True)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    model.zero_grad(set_to_none=True)
    loss.backward() 
    
    measurements = []
    for l in tqdm.tqdm(range(0, model.num_layers+1)):
        w = model.layers[f'fc_{l}'].weight
        w_grad = model.layers[f'fc_{l}'].weight.grad
        w_grad_fro = torch.linalg.matrix_norm(w_grad, ord='fro').item()
        w_ig = isometry_gap(w).item()
        fc_ig = isometry_gap(model.hiddens[f'fc_{l}']).item()
        if l < model.num_layers:
            act_ig = isometry_gap(model.hiddens[f'act_{l}']).item()
            norm_ig = isometry_gap(model.hiddens[f'norm_{l}']).item()
        else:
            # Final layer
            act_ig = np.nan
            norm_ig = np.nan
        measurements.append({
            'layer': l,
            'epoch': epoch,
            'weight_isogap': w_ig,
            'fc_isogap': fc_ig,
            'act_isogap': act_ig,
            'norm_isogap': norm_ig,
            'grad_fro_norm': w_grad_fro,
        })

    # Sanity cleaning gradients
    model.zero_grad(set_to_none=True)
    model.set_save_hidden(False)
    return measurements
    

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    
    pbar = tqdm.tqdm(loader, leave=False)
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_accuracy = correct / total
        
        pbar.set_description(f"train loss = {loss.item(): .4f}, train acc = {train_accuracy: .4f}")
        running_loss += loss.item()

        if math.isnan(running_loss):
            print("Train loss is nan", flush=True, file=sys.stderr)
            exit(1)
            
    train_accuracy = correct / total
    train_loss = running_loss / len(loader)
    return train_loss, train_accuracy

def test_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0.0
    total = 0.0 
    
    with torch.no_grad():
        pbar = tqdm.tqdm(loader, leave=False)
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1) 
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            test_accuracy = correct / total
            pbar.set_description(f"test loss = {loss.item(): .4f}, test acc = {test_accuracy: .4f}")
            
            if math.isnan(running_loss):
                print("Test loss is nan", flush=True, file=sys.stderr)
                exit(1)
                
    test_accuracy = correct / total 
    test_loss = running_loss / len(loader)
    return test_loss, test_accuracy
