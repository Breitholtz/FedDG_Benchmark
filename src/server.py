import copy
import time

from multiprocessing import pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict
import torch.distributions as dist

from .models import *
from .utils import *
from .client import *
from .dataset_bundle import *

import wandb

class FedAvg(object):
    def __init__(self, device, ds_bundle, hparam):
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.hparam = hparam
        self.num_rounds = hparam['num_rounds']
        self.fraction = hparam['fraction']
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.label_dist=None
        self.target_label_dist=None
        self.featurizer = None
        self.classifier = None
        self.client_labeldists=[]
    def setup_model(self, model_file=None, start_epoch=0):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        
        for client in tqdm(self.clients):
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
    
    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)

    
    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())

    def sample_clients(self):
        """
        Description: Sample a subset of clients. 
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    

    def update_clients(self, sampled_client_indices):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(self._round)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size


    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        for idx in tqdm(sampled_client_indices):
            self.clients[idx].client_evaluate()

    

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

    def compute_gradient_diff(self,sampled_client_indices):
        """Calculate the gradient difference in cosine similarity between clients"""
        from sklearn.metrics.pairwise import cosine_similarity
        #### gradients calc
        grads=comp_grads(client_models, global_model,lr) ## lr necessary?
        if len(old_grads)==0:
            pass
        else:
            # compute the difference between the vectors
            grad_diff=[]
            for i, grad in enumerate(grads):
                d=cosine_similarity(old_grads[i].reshape(1,-1),grad.reshape(1,-1))
                print(d)
                grad_diff.append(d)
            gradient_differences.append(grad_diff)
        old_grads=grads
        print("Hi")
        return
    def distance_to_global(self,sampled_client_indices):
        ## distance calc
        M=comp_dist(client_models)
        dist=np.triu(M,1).sum()
        client_dists.append(dist)
        print('The sum of the distances between clients: %0.3f' % dist)
        ## compute the distance of the clients from the global model
        client2global=dist_from_global(client_models,global_model)
        client_to_global.append(client2global)
        print('The distance from the clients to the global model before aggregation:', client2global)
    def estimate_label_dist(self, sampled_client_indices):
        ## we just return a list of the distribution tensors for the clients
        
        print(self.model.state_dict()['module.0.network.fc.weight'])
        for idx in sampled_client_indices:
            print(self.clients[idx].state.dict()['module.0.network.fc.weight'])

        # TODO: here the actual estimation should be
        
        print("The estimation should be here!")
        A
        
            
        ## For contrast, RLU needs : Auxiliary dataset A, the global model θ, local updates ∆θ_k ,∆W(t) , ∆b, learning rate η
        ## what should we do here? try to find a way to keep a running estimate of the label distribution
    
    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # here we should measure diff between global and local models, or down below before the aggregation
        
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # maybe estimate label dist here?
        if self.client_labeldists != None and self.hparam['dists_known']==1:
            pass # this has already been found
            #dists=self.client_labeldists
        else:
            #we have to estimate the label distributions
            dists=estimate_label_dist(sampled_client_indices)
        
        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
    
    def evaluate_global_model(self, dataloader, initial_dist=0):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self._featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1]/2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:,:z_dim]
                        z_sigma = F.softplus(features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(-1, features_params.shape[-1])
                        z_mu = flattened_features_params[:,:z_dim]
                        z_sigma = F.softplus(flattened_features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    features = self.featurizer(data)
                prediction = self.classifier(features)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
                # print("DEBUG: server.py:183")
                # break
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            #print(metric)
            
            
            if initial_dist:
                self.target_label_dist=np.bincount(np.array(y_true.to("cpu")))
            if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric

    def fit(self):
        import pickle #for saving result dict
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))

            self.train_federated_model()
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            for name, dataloader in self.test_dataloader.items():
                metric, result_str = self.evaluate_global_model(dataloader)
                metric_dict[name] = metric
                #print(metric_dict)
                #print(name)
                if name == 'val':
                    print(self.ds_bundle.key_metric)
                    lodo_val = metric[self.ds_bundle.key_metric]
                    if lodo_val > best_lodo_val_value:
                        best_lodo_val_round = r
                        best_lodo_val_value = lodo_val
                        lodo_flag = True
                if name == 'id_val':
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == 'test':
                    t_val = metric[self.ds_bundle.key_metric]
                if name == 'id_test':
                    id_t_val = metric[self.ds_bundle.key_metric]
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val
            
            print(metric_dict)
            if self.hparam['result_path']:
                
                # save/append the metrics dict to a file
                if self.hparam["server_method"]=="FedCustomWeights":
                    dirname=f"{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['server_method']}_{self.hparam['iid']}_{self.hparam['imbalanced_split']}_{self.hparam['custom_weighting']}_{self.hparam['reg_lambda']}"
                else:
                    dirname=f"{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['server_method']}_{self.hparam['iid']}_{self.hparam['imbalanced_split']}"
                filename=f"{r}_{self.hparam['seed']}_{'results.pkl'}" 
                if not os.path.exists(self.hparam['result_path']+str(dirname)):
                    os.makedirs(self.hparam['result_path']+str(dirname))
                with open(self.hparam['result_path']+str(dirname)+"/"+str(filename), 'wb') as f:
                    pickle.dump(metric_dict, f)
                
            if self.hparam['wandb']:
                wandb.log(metric_dict, step=self._round*self.hparam['local_epochs'])
            self.save_model(r)
            self._round += 1
        if self.hparam['wandb']:
            if best_id_val_round != 0: 
                wandb.summary['best_id_round'] = best_id_val_round
                wandb.summary['best_id_val_acc'] = best_id_val_test_value
            if best_lodo_val_round != 0:
                wandb.summary['best_lodo_round'] = best_lodo_val_round
                wandb.summary['best_lodo_val_acc'] = best_lodo_val_test_value
        else:
            print("best_id_round: " + str(best_id_val_round))
            print("best_id_val_acc: " + str(best_id_val_test_value))
            print("best_lodo_round: " + str(best_lodo_val_round))
            print("best_lodo_val_acc: " + str(best_lodo_val_test_value))
        self.transmit_model()

    def save_model(self, num_epoch):
        if self.hparam["server_method"]=="FedCustomWeights":
            if self.hparam["custom_weighting"]=="alphastar":
                path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['server_method']}_{self.hparam['custom_weighting']}_{self.hparam['iid']}_{self.hparam['imbalanced_split']}_{self.hparam['reg_lambda']}_{self.hparam['seed']}_{num_epoch}.pth"
            else:
                path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['server_method']}_{self.hparam['custom_weighting']}_{self.hparam['iid']}_{self.hparam['imbalanced_split']}_{self.hparam['seed']}_{num_epoch}.pth"
        else:
            path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['server_method']}_{self.hparam['iid']}_{self.hparam['imbalanced_split']}_{self.hparam['seed']}_{num_epoch}.pth"
        torch.save(self.model.state_dict(), path)

class FedCustomWeights(FedAvg):
 def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.reg_lambda=0
        self.alphastar=None
 def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        sum=0
        for idx in tqdm(sampled_client_indices):
            sum+=self.clients[idx].client_evaluate()
        return sum
 def aggregate(self, sampled_client_indices, coefficients, weighting='uniform', client_label_dists=None):
        """Average the updated and transmitted parameters from each selected client."""
        num_sampled_clients = len(sampled_client_indices)

        if self.hparam:
            weighting=self.hparam["custom_weighting"]
        
        averaged_weights = OrderedDict()
        if weighting=='performance':
            client_sum=self.evaluate_clients(sampled_client_indices)
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):

            local_weights = self.clients[idx].model.state_dict()
            #print(local_weights)
            if weighting=='uniform':
                ## just take 1/n where n is the number of clients
                coefficients=np.ones(np.array(coefficients).shape)/num_sampled_clients
            elif weighting=='random':
                ## take some random convex combination; changes every iteration
                #print(np.array(coefficients).shape)
                random_weights=np.random.rand(*np.array(coefficients).shape)
                S=np.sum(random_weights)
                coefficients=random_weights/S
                #print(coefficients)
            elif weighting=='performance':
                ## weighted based on client performance on validation(right now on local dataset)
                # calculate how much weight each client should have based on the measured accuracy
                coefficients=np.zeros(np.array(coefficients).shape)
                ### could just do for the idx client here
                #for i, ix in enumerate(self.clients):
                coefficients[it]= self.clients[idx].accuracy/client_sum
            elif weighting=='alphastar':
                self.reg_lambda = float(self.hparam['reg_lambda'])
                import scipy
                #print("We are in alphastar territory")
                ## here we want to do the minimization that finds the trade-off between clients and target while keeping sample efficicency in mind
                # min_alpha |T(y)-\sum_i alpha_i S_i(y)| + lambdasum_i (alpha_i/n_i)^2
                #print("Target labels: ",self.target_label_dist)
                #print("Client labels: ", self.label_dist)
                x0=np.ones(np.array(coefficients).shape)/num_sampled_clients # start at uniform
                def loss(alpha, target_labels=[], client_labels=[], reg_lambda=0):
                    sum=0
                    alpha_sum=0
                    for idx, label in enumerate(client_labels):
                        sum+=alpha[idx]*label
                        alpha_sum+=alpha[idx]**2/(np.sum(label))**2

                    obj=np.linalg.norm(target_labels-sum)+ alpha_sum*reg_lambda
                    return obj
                alphastar=scipy.optimize.minimize(loss, x0, args=(self.target_label_dist,self.label_dist, self.reg_lambda), constraints=(scipy.optimize.LinearConstraint(np.ones(len(x0)), ub=1)))
                self.alphastar=alphastar.x
                coefficients =alphastar.x
                print("Alpha*: ",coefficients)
            
            ## TODO: do one based on client coherence?
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    if averaged_weights[key].get_device()!=0:
                        averaged_weights[key]=averaged_weights[key].cuda()
                    if local_weights[key].get_device()!=0:
                        local_weights[key]=local_weights[key].cuda()
                    #print(local_weights[key].get_device())
                    averaged_weights[key] += coefficients[it] * local_weights[key]   
        self.model.load_state_dict(averaged_weights)

# def evaluate_clients(self, sampled_client_indices):
#     def evaluate_single_client(selected_index):
#         self.clients[selected_index].client_evaluate()
#         return True
#     for idx in tqdm(sampled_client_indices):
#         self.clients[idx].client_evaluate()



# def aggregate(self, sampled_client_indices, coefficients):
#     """Average the updated and transmitted parameters from each selected client."""
#     averaged_weights = OrderedDict()
#     for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
#         local_weights = self.clients[idx].model.state_dict()
#         for key in self.model.state_dict().keys():
#             if it == 0:
#                 averaged_weights[key] = coefficients[it] * local_weights[key]
#             else:
#                 averaged_weights[key] += coefficients[it] * local_weights[key]
#     self.model.load_state_dict(averaged_weights)
    
class FedDG(FedAvg):
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
            client.set_amploader(self.amploader)
        super().register_clients(clients)
            
    def set_amploader(self, amp_dataset):
        self.amploader = amp_dataset


class FedADGServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.gen_input_size = int(hparam['hparam5'])

    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self._generator = GeneDistrNet(num_labels=self.ds_bundle.n_classes, input_size=self.gen_input_size, hidden_size=self._featurizer.n_outputs)
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.generator = nn.DataParallel(self._generator)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier), copy.deepcopy(self._generator))

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict(), self._generator.state_dict())

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            del message
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].update_model(self.model.state_dict(), self._generator.state_dict())
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            del message

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        del message

        averaged_weights = OrderedDict()
        averaged_generator_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            local_generator_weights = self.clients[idx].generator.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]                 
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]         
            for key in self.generator.state_dict().keys():
                if it == 0:
                    averaged_generator_weights[key] = coefficients[it] * local_generator_weights[key]
                    
                else:
                    averaged_generator_weights[key] += coefficients[it] * local_generator_weights[key]
        self.model.load_state_dict(averaged_weights)
        self.generator.load_state_dict(averaged_generator_weights)


class FedGMA(FedAvg):
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        num_sampled_clients = len(sampled_client_indices)
        delta = []
        sign_delta = ParamDict()
        self.model.to('cpu')
        last_weights = ParamDict(self.model.state_dict())
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            self.clients[idx].model.to('cpu')
            local_weights = ParamDict(self.clients[idx].model.state_dict())
            delta.append(coefficients[it] * (local_weights - last_weights))
            if it == 0:
                sum_delta = delta[it]
                sign_delta = delta[it].sign()
            else:
                sum_delta += delta[it]
                sign_delta += delta[it].sign()
                # if it == 0:
                #     averaged_weights[key] = coefficients[it] * local_weights[key]
                # else:
                #     averaged_weights[key] += coefficients[it] * local_weights[key]
        sign_delta /= num_sampled_clients
        abs_sign_delta = sign_delta.abs()
        # print(sign_delta[key])
        mask = abs_sign_delta.ge(self.hparam['hparam1'])
        # print("--mid--")
        # print(mask)
        # print("-------")
        final_mask = mask + (0-mask) * abs_sign_delta
        averaged_weights = last_weights + self.hparam['hparam1'] * final_mask * sum_delta 
        self.model.load_state_dict(averaged_weights)



class ScaffoldServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.c = None

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
                client.c_global = copy.deepcopy(self.c)
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())
                self.clients[idx].c_global = copy.deepcopy(self.c)
    
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            if it == 0:
                c_local = self.clients[idx].c_local
            else:
                c_local += self.clients[idx].c_local
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
    
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.c = c_local / len(sampled_client_indices)
        self.model.load_state_dict(averaged_weights)


class AFLServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.group_weights = torch.zeros(self.ds_bundle.grouper.n_groups)
        train_set = self.ds_bundle.dataset.get_subset('train', transform=self.ds_bundle.train_transform)
        train_g = self.ds_bundle.grouper.metadata_to_group(train_set.metadata_array)
        unique_groups, unique_counts = torch.unique(train_g, sorted=False, return_counts=True)
        counts = torch.zeros(self.ds_bundle.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.is_group_in_train = is_group_in_train
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
       

    def transmit_lambda(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in tqdm(self.clients, leave=False):
            # for client in self.clients:
            
                client.update_vector(self.group_weights)
        else:
            # send the global model to selected clients
            for idx in tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_vector(self.group_weights)

    def aggregate(self,sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

    def update_lambda(self, sampled_client_indices):
        self.transmit_model(sampled_client_indices)
        total_loss_per_domain = torch.zeros_like(self.group_weights)
        total_samples_per_domain = torch.zeros_like(self.group_weights)
        # for client in tqdm(self.clients, leave=False):
        # # for client in self.clients:
        #     loss_per_domain, samples_per_domain = client.gradient_lambda()
        #     total_loss_per_domain += loss_per_domain
        #     total_samples_per_domain += samples_per_domain

        # send the global model to selected clients
        for idx in tqdm(sampled_client_indices, leave=False):
        # for idx in sampled_client_indices:
            loss_per_domain, samples_per_domain = self.clients[idx].gradient_lambda()
            total_loss_per_domain += loss_per_domain
            total_samples_per_domain += samples_per_domain
        self.group_weights += torch.nan_to_num(self.hparam['hparam1'] * total_loss_per_domain / total_samples_per_domain, nan=0.0)

        print(self.group_weights)

        self.group_weights = euclidean_proj_simplex(self.group_weights)

                
        print("after proj")
        print(self.group_weights)
        # print(self.group_weights)
        wandb.log({"l0_lmda": torch.count_nonzero(self.group_weights[self.group_weights>0.001])} ,step=self._round*self.hparam['local_epochs'])

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        self.transmit_lambda(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)

        self.update_lambda(sampled_client_indices)
