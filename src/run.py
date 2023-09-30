import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime, timezone
from modules.models import MLPWithBatchNorm, CustomNormalization
from modules.utils import train_one_epoch, test_one_epoch, isometry_gap, get_measurements
from modules.data_utils import TensorDataLoader, dataset_to_tensors
import time
import os
import json
from constants import *
from pprint import pprint


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--uid', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--num-layers', type=int, required=False)
    parser.add_argument('--hidden-dim', type=int, required=False)
    parser.add_argument('--batch-size', type=int, required=False)
    parser.add_argument('--init-type', type=str, required=False)
    parser.add_argument('--norm-type', type=str, required=False)
    parser.add_argument('--activation', type=str, required=False)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--order', type=str)
    parser.add_argument('--bias', type=int)
    parser.add_argument('--mean-reduction', type=int)
    parser.add_argument('--gain-exponent', type=float)
    parser.add_argument('--force-factor', type=float)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--save-every-epochs', type=int, default=10)
    args = parser.parse_args()
    # Get time
    now = datetime.now(timezone.utc)
    date_time = now.strftime("%Y_%m_%dT%H_%M_%SZ")

    # Make config
    config = vars(args)
    config['gain_exponent'] = -config['gain_exponent']
    config['bias'] = False if args.bias == 0 else True
    config['force_factor'] = args.force_factor if args.force_factor is not None else None
    config['mean_reduction'] = False if args.mean_reduction == 0 else True

    if config['norm_type'] == 'torch_bn':
        print("Using norm type torch. Ignoring mean reduction parameter and force factor parameter.", flush=True)

    # Make save directory
    config_path = os.path.join(os.environ['EXPERIMENT_DIR'], f"config_{config['uid']}.json")     
    experiment_path = os.path.join(os.environ['EXPERIMENT_DIR'], f"{config['uid']}")
    start_epoch = 0

    # If uid exists, then restart from checkpoint
    if os.path.exists(config_path) and os.path.exists(experiment_path):
        print("Restarting from checkpoint.", flush=True)
        files = os.listdir(experiment_path)
        epoch = max([int(x.split('_')[-1][:-4]) for x in files if x.endswith(".pth")])
        start_epoch = epoch
        print(f"Starting from epoch: {start_epoch}", flush=True)
        config = json.loads(open(config_path, 'r').read())
        print("Loaded config.")
    else: 
        os.mkdir(experiment_path)
        config.update({
            'datetime': date_time,
            'input_dim': DS_INPUT_SIZES[config['dataset']],
            'output_dim': DS_NUM_CLASSES[config['dataset']], 
        })

        json.dump(config, open(config_path, 'w'), indent=4)
        
    pprint(config, indent=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get dataset directory
    ds = getattr(torchvision.datasets, config['dataset'])
    transform = DS_TRANSFORMS[config['dataset']]
    trainset = ds(root=os.environ['DATASET_DIR'], train=True, download=False, transform=transform)
    testset = ds(root=os.environ['DATASET_DIR'], train=False, download=False, transform=transform)
    
    trainloader = TensorDataLoader(*dataset_to_tensors(trainset, device=device), batch_size=config['batch_size'], shuffle=True)
    testloader = TensorDataLoader(*dataset_to_tensors(testset, device=device), batch_size=config['batch_size'], shuffle=False)

    measurements_inputs, measurements_labels = next(iter(trainloader))

    # Optimizers
    model = MLPWithBatchNorm(input_dim=config['input_dim'], 
                             output_dim=config['output_dim'], 
                             num_layers=config['num_layers'], 
                             hidden_dim=config['hidden_dim'],
                             norm_type=config['norm_type'],
                             mean_reduction=config['mean_reduction'],
                             activation=ACTIVATIONS[config['activation']],
                             save_hidden=False,
                             exponent=config['gain_exponent'],
                             order=config['order'],
                             force_factor=config['force_factor'],
                             bias=config['bias']).to(device)
    model.reset_parameters(config['init_type'])

    if start_epoch != 0:
        model.load_state_dict(torch.load(os.path.join(experiment_path, f"model_epoch_{start_epoch}.pth")))
        print(f"Loaded model checkpoint at epoch {start_epoch}", flush=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    df = []
    if start_epoch != 0:
        df = pd.read_csv(os.path.join(experiment_path, f'results_epoch_{start_epoch}.csv')).drop(columns=['Unnamed: 0']).to_dict('records')

    if start_epoch == 0:
        # Measure all at init
        test_loss, test_acc = test_one_epoch(model, testloader, criterion, device)
        train_loss, train_acc = test_one_epoch(model, trainloader, criterion, device)
        df.append({
                'epoch': 0,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
            })
        try:
            measurements = get_measurements(model, measurements_inputs, measurements_labels, criterion, 0, device)
            temp_df = pd.DataFrame(measurements)
            temp_df.to_csv(os.path.join(experiment_path, 'measurements_epoch_0.csv'))
        except RuntimeError:
            pass
        ########################################

    start_epoch += 1 # increase epoch
    # Start training loop
    for epoch in range(start_epoch, config['num_epochs']+1):
        start = time.time()
        print(f"Epoch: {epoch}/{config['num_epochs']+1}", flush=True)

        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = test_one_epoch(model, testloader, criterion, device)
        
        df.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc,
        })

        # Log dataframe after every epoch
        temp_df = pd.DataFrame(df)
        temp_df.to_csv(os.path.join(experiment_path, f'results_epoch_{epoch}.csv')) 
        
        try:
            measurements = get_measurements(model, measurements_inputs, measurements_labels, criterion, epoch, device)
            temp_df = pd.DataFrame(measurements)
            temp_df.to_csv(os.path.join(experiment_path, f'measurements_epoch_{epoch}.csv')) 
        except RuntimeError:
            pass 
        if epoch % config['save_every_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(experiment_path, f"model_epoch_{epoch}.pth"))
            print(f"Checkpointed to disk at epoch: {epoch}", flush=True) 
        end = time.time()

        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, test loss: {test_loss:.4f}, test_acc: {test_acc:.4f}", flush=True)        
        print(f"Time spent: {(end - start): .2f}s", flush=True)
