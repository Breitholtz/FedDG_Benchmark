import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import random
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
from pathlib import Path
from options import args_parser
import json
import copy

from models import *
from utils import *


if __name__ == '__main__':
    print("run hparam search")   
    args = args_parser()

    #read config file
    with open(args.config_file) as fh:
        config = json.load(fh)

    for key in config:
        setattr(args, key, config[key])

    print(args)
    
    if(args.seed == None):
        seed = random.randint(1,10000)
    else:
        seed = args.seed
        
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    count = 0
    experiment_dir = f"./save/hparam/{args.dataset}/{args.dataset}_{seed}_{count}"
    while os.path.exists(experiment_dir):
        count=count+1
        experiment_dir = f"./save/hparam/{args.dataset}/{args.dataset}_{seed}_{count}"

    os.mkdir(experiment_dir)

    filename = 'results_hparam'
    filexist = os.path.isfile(f"./save/{args.dataset}/"+filename) 
    if(not filexist):
        with open(f"./save/{args.dataset}/"+filename,'a') as f1:

            f1.write('n_rounds;num_clients;local_ep;bs;lr;lr_alpha;seed;dirichlet_beta;method;dataset;val_acc_fedavg;val_acc_fedalpha;val_loss_fedavg;val_loss_fedalpha')

            f1.write('\n')

    #save config file
    with open(experiment_dir+'/config.json', 'w') as fh:
        json.dump(config, fh)

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if(args.dataset=='mnist'):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
        #split into train and val
        n_train = int(len(dataset)*0.8)
        n_val = len(dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_targets = extract_targets(train_dataset, dataset)
        train_dataset.targets = train_targets

        #create validation loader
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)
        origin_model = MLP_MNIST(784, 64, 10)

    n_samples_per_client = np.random.dirichlet([args.dirichlet_beta] * args.num_clients, 1)[0]
    n_samples_per_client = np.round(n_samples_per_client/np.sum(n_samples_per_client)*len(train_dataset))
    n_samples_per_client = n_samples_per_client.astype(int)
    client_datasets, client_loaders = generate_federated_datasets(train_dataset, args.num_clients, args.dirichlet_beta, n_samples_per_client, args.bs)

    plot_label_distributions(client_loaders,save_path = experiment_dir+'/label_distributions.png')

    source_label_list = [get_label_distribution(client_datasets[i],n_labels=10) for i in range (len(client_datasets))]

    source_labels = np.array([])
    for s in source_label_list:
        #create matrix of all source label distributions:
        source_labels = np.append(source_labels,s)
    source_labels = source_labels.reshape(len(client_datasets),10)
    source_labels = torch.tensor(source_labels).T

    model_fed = copy.deepcopy(origin_model)
    criterion = nn.CrossEntropyLoss()
    print('Number of clients: ', args.num_clients)
    fed_val_loader = None

    if(args.method == 'fedavg'):
        val_acc_fedalpha, val_loss_fedalpha = [np.nan], [np.nan]
        clients_fed = []
        for i in range(args.num_clients-1): #use last client as test client
            clients_fed.append(FederatedClient(copy.deepcopy(origin_model), criterion, client_loaders[i], device, fed_val_loader))

        print(len(clients_fed))
        model_fedavg, train_loss_fedavg, train_acc_fedavg, test_acc_fedavg, val_loss_fed, val_acc_fed = train_fed(args.n_rounds, args.local_ep, clients_fed, args.lr, args.lr_alpha, val_loader, test_loader=client_loaders[-1], wd=args.wd, optimize_alpha_bool=False, target_labels=None, source_labels=None)

    if(args.method == 'fedalpha'):
        val_acc_fed, val_loss_fed = [np.nan], [np.nan]
        clients_fed = []
        for i in range(args.num_clients-1):
            clients_fed.append(FederatedClient(copy.deepcopy(origin_model), criterion, client_loaders[i], device, val_loader))
        print(len(clients_fed))
        model_fedalpha, train_loss_fedalpha, train_acc_fedalpha, test_acc_fedalpha, val_loss_fedalpha, val_acc_fedalpha = train_fed(args.n_rounds, args.local_ep, clients_fed, args.lr, args.lr_alpha, val_loader, test_loader=client_loaders[-1], wd=args.wd, optimize_alpha_bool=True, target_labels=source_labels[:,-1], source_labels=source_labels[:,0:-1])


    with open(f"./save/{args.dataset}/"+filename,'a') as f1:
        f1.write(f'{args.n_rounds};{args.num_clients};{args.local_ep};{args.bs};{args.lr};{args.lr_alpha};{seed};{args.dirichlet_beta};{args.method};{args.dataset};{val_acc_fed[-1]};{val_acc_fedalpha[-1]};{val_loss_fed[-1]};{val_loss_fedalpha[-1]}')
        f1.write('\n')


    #save train_loss, train_acc, val_acc, val_loss
    if(args.method == 'fedalpha'):
        np.save(experiment_dir+'/train_loss_fedalpha.npy',train_loss_fedalpha)
        np.save(experiment_dir+'/train_acc_fedalpha.npy',train_acc_fedalpha)
        np.save(experiment_dir+'/val_acc_fedalpha.npy',val_acc_fedalpha)
        np.save(experiment_dir+'/val_loss_fedalpha.npy',val_loss_fedalpha)
        torch.save(model_fedalpha.state_dict(), experiment_dir+'/model_fedalpha.pth')

    if(args.method == 'fedavg'):
        np.save(experiment_dir+'/train_loss_fedavg.npy',train_loss_fedavg)
        np.save(experiment_dir+'/train_acc_fedavg.npy',train_acc_fedavg)
        np.save(experiment_dir+'/val_acc_fedavg.npy',val_acc_fed)
        np.save(experiment_dir+'/val_loss_fedavg.npy',val_loss_fed)
        torch.save(model_fedavg.state_dict(), experiment_dir+'/model_fedavg.pth')
            
                