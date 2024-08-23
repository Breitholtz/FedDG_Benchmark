import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import random
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import functional as F
import scipy
from collections import defaultdict

class FederatedClient():
    def __init__(self, model, criterion, train_loader, device, val_loader=None):
        self.model = model
        #self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def train(self, num_epochs, learning_rate, weight_decay, fed_alg, mu=0):
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model.to(self.device)
        self.model.train()
        global_model = copy.deepcopy(self.model)
        train_loss = []
        train_acc = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_acc = 0
            loss1_sum = 0
            loss2_sum = 0
            for i, (x, y) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                if(fed_alg=='fedavg'):
                    loss = self.criterion(outputs, y)
                elif(fed_alg=='fedprox'):
                    proximal_term = 0.0
                    for param, global_param in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += torch.norm(param - global_param, p=2)
                    loss1 = self.criterion(outputs, y) 
                    loss2 = mu/2 * proximal_term
                    loss = loss1 + loss2
                loss.backward()

                # Clip gradients
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

                optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += (outputs.argmax(1) == y).sum().item()

            train_loss.append(epoch_loss / len(self.train_loader))
            train_acc.append(epoch_acc / len(self.train_loader.dataset))
        
        return self.model, train_loss, train_acc
    
    def test(self,test_loader):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                test_loss += loss.item()
                test_acc += (outputs.argmax(1) == y).sum().item()
        
        return test_loss / len(test_loader), test_acc / len(test_loader.dataset)
    
    def get_model(self):
        return self.model

def FedAvg(w,alpha):
    #alpha = alpha/np.sum(alpha) #normalize alpha
    w_avg = copy.deepcopy(w[0])
    n_clients = len(w)
    
    for l in w_avg.keys():
        w_avg[l] = w_avg[l] - w_avg[l]

    for l, layer in enumerate(w_avg.keys()): #for each layer
        w_kl = []
        for k in range(0,n_clients): #for each client
            w_avg[layer] += alpha[k]*w[k][layer]
    return w_avg
def softmax(x,tau):
    #x is a vector, tau is the temperature. large tau -> uniform distribution, small tau -> argmax
    return torch.exp(x/tau)/torch.sum(torch.exp(x/tau),dim=0)

def alpha_loss(alpha, target_labels=[], source_labels=[], dataset_sizes =[], reg_lambda=0):
    obj = np.linalg.norm(target_labels - np.matmul(source_labels,alpha),ord=2)**2 + reg_lambda * np.linalg.norm(alpha,ord=2)**2/np.linalg.norm(dataset_sizes,ord=1)
    return obj

def optimize_alpha(target_labels, source_labels, alpha, dataset_sizes, reg_lambda):
    #given a vector T(y) (target labels) and S(y) (source labels), optimize alpha with SGD
    #initialize alpha
    dataset_sizes = torch.tensor(dataset_sizes)
    #alpha = torch.tensor(alpha, requires_grad=True)
    #optimizer = optim.SGD([alpha], lr=eta)
    #for i in range(n_epochs):
        #soft_alpha = softmax(alpha, 0.1)
        #loss = torch.norm(target_labels - torch.matmul(source_labels,alpha),p=2)
        #loss2 = mu * torch.norm(alpha/dataset_sizes,p=2)
        #loss = loss1 + loss2
        #optimizer.zero_grad()
        #loss.backward()

        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_([alpha], 0.5)

        #optimizer.step()
        
        #print(alpha.detach().numpy(), loss.item())
    constraints = scipy.optimize.LinearConstraint(np.ones(len(alpha)), lb=1, ub=1)
    bounds = scipy.optimize.Bounds(0,1)
    alpha = scipy.optimize.minimize(alpha_loss, alpha, args=(target_labels,source_labels, dataset_sizes, reg_lambda), constraints=constraints, bounds=bounds)
    return alpha

def train_fed(n_communication, num_local_epochs, clients_fed, lr, lr_alpha, val_loader, test_loader, wd=0.0, optimize_alpha_bool=False, target_labels=None, source_labels=None, reg_lambda=0.0):
    n_clients = len(clients_fed)
    mean_loss_fed = []
    mean_acc_fed = []
    val_loss_fed = []
    test_acc_fed, val_acc_fed = [], []
    dataset_sizes = [clients_fed[i].train_loader.dataset.__len__() for i in range(n_clients)]
    n_early_stopping = 105
    count = 0
    best_val_loss = np.inf
    best_model = copy.deepcopy(clients_fed[0].model)
    if(not optimize_alpha_bool):
        alpha = np.array(dataset_sizes)/np.sum(dataset_sizes)
        print(alpha)
    for k in range(n_communication):
        train_losses_fed = []
        train_accs_fed = []
        param = []
        for i in range(n_clients-1):
            client_model, train_loss_fed, train_acc_fed = clients_fed[i].train(num_local_epochs,lr,wd,'fedavg')
            param.append(copy.deepcopy(client_model.state_dict()))
            train_losses_fed.append(train_loss_fed[-1])
            train_accs_fed.append(train_acc_fed[-1])

        mean_loss_fed.append(np.mean(train_losses_fed))
        mean_acc_fed.append(np.mean(train_accs_fed))
        print(f'Round {k} - mean loss: {mean_loss_fed[-1]}, mean acc: {mean_acc_fed[-1]}')

        if(optimize_alpha_bool):
            print(f'Round {k}')
            #initialize alpha
            alpha = np.random.rand(n_clients)
            alpha = alpha/np.sum(alpha)
            alpha = optimize_alpha(target_labels, source_labels, alpha, dataset_sizes, reg_lambda)
            alpha = alpha.x
            #alpha = alpha/np.sum(alpha)
        w_global_model_fedavg = FedAvg(param, alpha)
        for i in range(n_clients):
            clients_fed[i].model.load_state_dict(copy.deepcopy(w_global_model_fedavg))


        val_loss, val_acc = clients_fed[0].test(val_loader)
        val_acc_fed.append(val_acc)
        val_loss_fed.append(val_loss)
        if(val_loss < best_val_loss):
            best_model = copy.deepcopy(clients_fed[0].model)
            best_val_loss = val_loss
            count = 0
        else:
            count += 1
        if(count == n_early_stopping):
            print(f'Early stopping at round {k}')
            break

        if(test_loader is not None):
            test_loss, test_acc = clients_fed[0].test(test_loader)
            test_acc_fed.append(test_acc)
        

    return best_model, mean_loss_fed, mean_acc_fed, test_acc_fed, val_loss_fed, val_acc_fed


def generate_federated_datasets(dataset, num_clients, alpha, num_samples_per_client, bs):
    """
    Generate federated datasets using Dirichlet distribution.
    
    :param dataset: PyTorch dataset (e.g., MNIST)
    :param num_clients: Number of clients
    :param alpha: Concentration parameter for Dirichlet distribution
    :param num_samples_per_client: List of number of samples per client
    :return: List of DataLoaders for each client
    """
    assert len(num_samples_per_client) == num_clients, "Length of num_samples_per_client must match num_clients"
    
    num_classes = len(np.unique(dataset.targets))
    
    # Generate Dirichlet distribution for label distributions for each client
    label_distributions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Get indices of each class in the dataset
    class_indices = [np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)]
    
    # Allocate instances to each client based on the generated label distributions
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.cumsum(label_distributions[c])
        proportions = np.insert(proportions, 0, 0)
        class_splits = [int(p * len(class_indices[c])) for p in proportions]
        
        for i in range(num_clients):
            client_indices[i].extend(class_indices[c][class_splits[i]:class_splits[i+1]])
    
    # Create DataLoader for each client
    client_loaders = []
    client_datasets = []
    for i, indices in enumerate(client_indices):
        if num_samples_per_client[i] < len(indices):
            indices = np.random.choice(indices, num_samples_per_client[i], replace=False)
        client_dataset = Subset(dataset, indices)
        client_loaders.append(DataLoader(client_dataset, batch_size=bs, shuffle=True))
        client_datasets.append(client_dataset)
    
    return client_datasets, client_loaders

#plot a heatmap: X-axis is client number, Y-axis is label number, and the value in each cell is the number of samples in that client with that label.
def plot_label_distributions(client_loaders, save_path=None):
    num_clients = len(client_loaders)
    num_classes = len(np.unique(client_loaders[0].dataset.dataset.targets))
    label_counts = np.zeros((num_clients, num_classes))
    
    for i, loader in enumerate(client_loaders):
        for _, labels in loader:
            for c in labels:
                label_counts[i][c] += 1
    
    #plt.figure(figsize=(5, 8))
    sns.heatmap(label_counts.T, annot=True, fmt=".1f", cmap="Blues", cbar=False)
    plt.xlabel("Client ID")
    plt.ylabel("Label")
    plt.title("Label distributions for each client")
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


# Function to split the dataset into N clients with only two labels each
def split_labels_among_clients(dataset, num_clients):
    labels = np.array(dataset.targets)
    label_indices = defaultdict(list)
    
    # Store indices of each label
    for idx, label in enumerate(labels):
        label_indices[label.item()].append(idx)
    
    client_data_indices = [[] for _ in range(num_clients)]
    label_pool = list(label_indices.keys())
    
    # Distribute two labels to each client
    for i in range(num_clients):
        chosen_labels = np.random.choice(label_pool, 2, replace=False)
        for label in chosen_labels:
            client_data_indices[i].extend(label_indices[label])
        
    # Create datasets for each client
    client_datasets = [Subset(dataset, indices) for indices in client_data_indices]
    
    return client_datasets

def split_data(dataset,n_parts):
    #split dataset into equal parts uniformly
    dataset_length = len(dataset)
    splits = np.array_split(np.arange(dataset_length), n_parts)
    dataset_sizes = [len(split) for split in splits]
    client_datasets = torch.utils.data.random_split(dataset, lengths=dataset_sizes)
    return client_datasets

def get_label_distribution(dataset,n_labels):
    #get the distribution of labels in a dataset and return a vector with the fraction of samples for each label
    labels = []
    for data in dataset:
        labels.append(data[1])
    labels = np.array(labels)
    label_distribution = np.zeros(n_labels)
    for i in range(n_labels):
        label_distribution[i] = np.sum(labels==i)
    label_distribution = label_distribution/np.sum(label_distribution)
    return label_distribution

# Function to extract targets
def extract_targets(subset, original_dataset):
    return [original_dataset.targets[i] for i in subset.indices]