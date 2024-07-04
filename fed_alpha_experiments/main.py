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

from models import *
from utils import *

if __name__ == '__main__':   
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
    experiment_dir = f"./save/{args.dataset}_{seed}_{count}"
    while os.path.exists(experiment_dir):
        count=count+1
        experiment_dir = f"./save/{args.dataset}_{seed}_{count}"

    os.mkdir(experiment_dir)


    filename = 'results'
    filexist = os.path.isfile(experiment_dir+'/'+filename) 
    if(not filexist):
        with open(experiment_dir+'/'+filename,'a') as f1:

            f1.write('n_rounds;num_clients;local_ep;bs;lr;seed;dirichlet_beta;dataset;test_acc_fedavg;test_acc_fedalpha')

            f1.write('\n')

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if(args.dataset=='mnist'):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
        origin_model = MLP_MNIST(784, 64, 10)

    n_samples_per_client = np.random.dirichlet([args.dirichlet_beta] * args.num_clients, 1)[0]
    n_samples_per_client = np.round(n_samples_per_client/np.sum(n_samples_per_client)*len(dataset))
    n_samples_per_client = n_samples_per_client.astype(int)
    client_datasets, client_loaders = generate_federated_datasets(dataset, args.num_clients, args.dirichlet_beta, n_samples_per_client, args.bs)

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
    val_loader = None

    clients_fed = []
    for i in range(args.num_clients-1): #use last client as test client
        clients_fed.append(FederatedClient(copy.deepcopy(origin_model), criterion, client_loaders[i], device, val_loader))

    print(len(clients_fed))

    model_fedavg, train_loss_fedavg, train_acc_fedavg, test_acc_fedavg = train_fed(args.n_rounds, args.local_ep, clients_fed, args.lr, test_loader=client_loaders[-1], wd=args.wd, optimize_alpha_bool=False, target_labels=None, source_labels=None)

    clients_fed = []
    for i in range(args.num_clients-1):
        clients_fed.append(FederatedClient(copy.deepcopy(origin_model), criterion, client_loaders[i], device, val_loader))
    print(len(clients_fed))
    model_fedalpha, train_loss_fedalpha, train_acc_fedalpha, test_acc_fedalpha = train_fed(args.n_rounds, args.local_ep, clients_fed, args.lr, test_loader=client_loaders[-1], wd=args.wd, optimize_alpha_bool=True, target_labels=source_labels[:,-1], source_labels=source_labels[:,0:-1])


    with open(experiment_dir+'/'+filename,'a') as f1:
        f1.write(f'{args.n_rounds};{args.num_clients};{args.local_ep};{args.bs};{args.lr};{seed};{args.dirichlet_beta};{args.dataset};{test_acc_fedavg[-1]};{test_acc_fedalpha[-1]}')
        f1.write('\n')

    sns.set_theme()
    plt.figure()
    plt.plot(train_loss_fedalpha)
    plt.plot(train_loss_fedavg)
    plt.xlabel('Communication rounds')
    plt.ylabel('Training loss')
    plt.legend(['FedAlpha', 'FedAvg'])
    plt.savefig(experiment_dir+'/train_loss.png')

    plt.figure()
    plt.plot(train_acc_fedalpha)
    plt.plot(train_acc_fedavg)
    plt.xlabel('Communication rounds')
    plt.ylabel('Train accuracy')
    plt.legend(['FedAlpha', 'FedAvg'])
    plt.savefig(experiment_dir+'/train_acc.png')

    plt.figure()
    plt.plot(test_acc_fedalpha)
    plt.plot(test_acc_fedavg)
    plt.xlabel('Communication rounds')
    plt.ylabel('Test accuracy')
    plt.legend(['FedAlpha', 'FedAvg'])
    plt.savefig(experiment_dir+'/test_acc.png')

    #save train_loss, train_acc, test_acc
    np.save(experiment_dir+'/train_loss_fedalpha.npy',train_loss_fedalpha)
    np.save(experiment_dir+'/train_acc_fedalpha.npy',train_acc_fedalpha)
    np.save(experiment_dir+'/test_acc_fedalpha.npy',test_acc_fedalpha)
    np.save(experiment_dir+'/train_loss_fedavg.npy',train_loss_fedavg)
    np.save(experiment_dir+'/train_acc_fedavg.npy',train_acc_fedavg)
    np.save(experiment_dir+'/test_acc_fedavg.npy',test_acc_fedavg)

    #save model
    torch.save(model_fedalpha.state_dict(), experiment_dir+'/model_fedalpha.pth')
    torch.save(model_fedavg.state_dict(), experiment_dir+'/model_fedavg.pth')