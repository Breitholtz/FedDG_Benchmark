from re import L
import torch
import copy
import numpy as np
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import ImageFolder
from wilds.datasets.wilds_dataset import WILDSSubset, WILDSDataset

class DatasetSplitter(object):
    """
    Returned dataset should be wilds dataset. We need to implement dataset transformer.
    All the dataset should not be dicts, but one dataset with x,y,metadata included.
    
    The output datasets should be dict contains dataset: 1) TensorDataset; 2)TransformSubset; 3)Wildssubset.
    """
    def split(self):
        pass



class ImbalancedSplitter(DatasetSplitter):
    def __init__(self, num_clients, config, test_classes, seed=1):
        self.test_classes = test_classes # which class should be missing
        ## what should be in here?, how skewed the split should be, config of the run,
        self.num_clients=num_clients
        self.config=config
        self.seed=seed
        self.rng = np.random.default_rng(self.seed)
        
    def split(self, dataset, domain_field, transform=None):
        # dataset: WILDSubset or WILDSDataset
        
        ## split should take into account how many clients there are, if more than the number of classes maybe we do not run
        ## the split thereafter puts many samples in one or two clients and few in the last one, the clients do not have access to a specific class each 
        if self.num_clients!=3:
            # this is not what this construction is for
            return NotImplementedError
        domain_field = dataset._metadata_fields.index(domain_field[0])
        num_examples_per_domain = np.bincount(dataset.metadata_array[:,domain_field])
        print("number of examples per domain:",num_examples_per_domain)
        #print(np.unique(np.array(dataset.y_array)))
        indices = np.array(dataset.indices)
        
        class1_idx= np.where(dataset.y_array == self.test_classes[0])[0] ## do this for all the different classes? share the rest equally?
        class2_idx= np.where(dataset.y_array == self.test_classes[1])[0]
        class3_idx= np.where(dataset.y_array == self.test_classes[2])[0]
        ## remove the ones which are not in the indices list?
        
        print("Class amounts: ", len(class1_idx),len(class2_idx),len(class3_idx))
        ## remove the odd classes from the total list
        rem_idx = indices
        class_indices=np.append(class1_idx,class2_idx)
        class_indices=np.append(class_indices,class3_idx)
        class_indices=np.ravel(class_indices)
        rem_idx=np.delete(rem_idx, class_indices)
        
        #split odd classes in two and give them to two clients; construct index sets for the three clients
        perm_indices = self.rng.permutation(rem_idx)

        # as a final step we make one of the clients sample size much smaller 10% of remaining data vs. 45% each for the other two, could make this more extreme
        pos1=int(0.33*len(perm_indices))
        pos2=int(0.66*len(perm_indices))
        
        client1=np.append(np.append(perm_indices[:pos1],class1_idx[:int(0.5*len(class1_idx))]),class2_idx[:int(0.5*len(class2_idx))]) ## not seen 3
        print("# client 1 before:",len(client1))
        #print(np.random.choice(client1, int(0.1*len(client1)),replace=False))
        client1=np.random.choice(client1, int(0.1*len(client1)),replace=False)
        print("# client 1 after:",len(client1))
        client2=np.append(np.append(perm_indices[pos1:pos2],class1_idx[int(0.5*len(class1_idx)):]),class3_idx[:int(0.5*len(class3_idx))]) ## not seen 2 
        client3=np.append(np.append(perm_indices[pos2:],class2_idx[int(0.5*len(class2_idx)):]),class3_idx[int(0.5*len(class3_idx)):]) ## not seen 1
        print("# client 2:",len(client2))
        print("# client 3:",len(client3))
         
        
        datasets=[]
        if isinstance(dataset, WILDSSubset):
            datasets.append(WILDSSubset(dataset.dataset, client1.tolist(), transform=transform))
            datasets.append(WILDSSubset(dataset.dataset, client2.tolist(), transform=transform))
            datasets.append(WILDSSubset(dataset.dataset, client3.tolist(), transform=transform))
        elif isinstance(dataset, WILDSDataset):
            datasets.append(WILDSSubset(dataset, client1.tolist(), transform=transform))
            datasets.append(WILDSSubset(dataset, client2.tolist(), transform=transform))
            datasets.append(WILDSSubset(dataset, client3.tolist(), transform=transform))
        else:
            return NotImplementedError

        return datasets
        
class LeaveOneDomainOutSplitter(DatasetSplitter):
    def __init__(self, test_domain):
        self.test_domain = test_domain
    
    def split(self, dataset, domain_field, transform=None):
        # dataset: WILDSubset or WILDSDataset
        indices = np.array(dataset.indices)
        tar_idx = np.where(dataset.metadata_array[:,domain_field] == self.test_domain)[0]
        rem_idx = np.where(dataset.metadata_array[:,domain_field] != self.test_domain)[0]
        if isinstance(dataset, WILDSSubset):
            return WILDSSubset(dataset.dataset, indices[tar_idx].tolist(), transform=transform), WILDSSubset(dataset.dataset, indices[rem_idx].tolist(), transform=transform)
        elif isinstance(dataset, WILDSDataset):
            return WILDSSubset(dataset, tar_idx.tolist(), transform=transform), WILDSSubset(dataset, rem_idx.tolist(), transform=transform)
        else:
            return NotImplementedError


class RandomSplitter(DatasetSplitter):
    def __init__(self, ratio, seed):
        self.ratio = ratio
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
    
    def split(self, dataset, transform=None):
        if isinstance(dataset, WILDSSubset):
            indices = np.array(dataset.indices)
            ori_dataset = dataset.dataset
        elif isinstance(dataset, WILDSDataset):
            indices = np.arange(len(dataset))
            ori_dataset = dataset
        else:
            return NotImplementedError
        
        perm_indices = self.rng.permutation(indices)
        pos = int(self.ratio * len(perm_indices))
        return WILDSSubset(ori_dataset, perm_indices[0:pos].tolist(), transform=transform), WILDSSubset(ori_dataset, perm_indices[pos:].tolist(), transform=transform)


class NonIIDSplitter():
    def __init__(self, num_shards, iid, seed):
        """
        num_shards: split dataset to clients
        iid: float from 0-1, describe how iid it is. iid=0 means each shard only contains images from one domain, iid=1 means each shard shares the same domain distribution.
        """
        self.num_shards = num_shards
        self.iid = iid
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def split(self, dataset, domain_field, transform=None):
        '''
        dataset: WILDSSubset
        '''            
        domain_field = dataset._metadata_fields.index(domain_field[0])
        num_examples_per_domain = np.bincount(dataset.metadata_array[:,domain_field]) # compute number of examples per domain.
        num_domains = len(num_examples_per_domain)                                # number of domains. Notice that we use bincount, that means the metaarray should be integer at least.
        non_empty_num_domains = sum(num_examples_per_domain > 0)
        if non_empty_num_domains <= self.num_shards:
            main_shards_per_domain = (num_examples_per_domain > 0).astype('int')          # initialize the data structure to store each domain‘s main shards 
            print(np.sum(main_shards_per_domain))
            while np.sum(main_shards_per_domain) < self.num_shards:                       # start to distribute domain to 
                ratio = np.divide(num_examples_per_domain.astype('float'), main_shards_per_domain.astype('float'), out=np.zeros_like(num_examples_per_domain.astype('float')), where=main_shards_per_domain!=0)
                argmax = np.argmax(ratio)
                main_shards_per_domain[argmax] += 1
            # print(main_shards_per_domain)
            main_domain_per_shards = []
            for i, num_shard in enumerate(main_shards_per_domain):
                main_domain_per_shards += [i] * int(num_shard)
            # print(main_domain_per_shards)
            # print(len(main_domain_per_shards))
            num_examples_per_shards = []
            main_domain_ratio_per_shard = np.array([1/u if u != 0 else 0 for u in main_shards_per_domain]) # len = num_domains
            non_main_domain_ratio_per_shard = 1/self.num_shards # len = num_domains
            for i, main_domain in enumerate(main_domain_per_shards):
                main_domain_onehot = np.zeros(len(main_shards_per_domain))
                main_domain_onehot[main_domain] = 1
                num_examples_per_shards.append(num_examples_per_domain * (main_domain_ratio_per_shard * main_domain_onehot * (1 - self.iid)
                                        + non_main_domain_ratio_per_shard * self.iid))
            np_examples_per_shards = np.array(num_examples_per_shards)
            np_int_examples_per_shards = np_examples_per_shards.astype(int)
            diff = np.rint(np.sum(np_examples_per_shards - np_int_examples_per_shards, axis=0)).astype(int)
            diff_mask = np.zeros((self.num_shards,num_domains))
            for col in range(num_domains):
                diff_mask[0:diff[col],col] = 1
                final_examples_per_shards = (np.rint(np_int_examples_per_shards + diff_mask)).astype(int)
            
        else:
            num_examples_per_shards_0 = np.zeros((self.num_shards, num_domains))
            desc_idx = np.argsort(num_examples_per_domain)[::-1]

            for idx in desc_idx:
                amin = np.argmin(np.sum(num_examples_per_shards_0, axis=1))
                num_examples_per_shards_0[amin, idx] = num_examples_per_domain[idx]
            
            num_examples_per_shards_1 = np.zeros((self.num_shards, num_domains))
            for shard in num_examples_per_shards_1:
                shard += num_examples_per_domain / self.num_shards
            float_final_examples_per_shards = num_examples_per_shards_1 * self.iid + num_examples_per_shards_0 * (1-self.iid)
            # could be float need to change to int.
            final_examples_per_shards = np.floor(float_final_examples_per_shards).astype('int')
            diff_array = (num_examples_per_domain - np.sum(final_examples_per_shards, axis=0)).astype('int')
            for i, diff in enumerate(diff_array):
                for d in range(diff):
                    final_examples_per_shards[d, i] += 1

            assert np.sum(num_examples_per_domain - np.sum(final_examples_per_shards, axis=0)) == 0
        indices_per_domain = [] # assert len(indices_per_domain) == num_domains
        for i in range(num_domains):
            sub_indices = np.where(dataset.metadata_array[:,domain_field] == i)[0]
            if isinstance(dataset, WILDSSubset):
                indices = np.array(dataset.indices)[sub_indices]
            elif isinstance(dataset, WILDSDataset):
                indices = sub_indices
            else:
                raise NotImplementedError
            perm_indices = self.rng.permutation(indices)
            indices_per_domain.append(perm_indices)

        dataset_per_shards = []
        pointer = np.zeros(num_domains, dtype=np.int64)
        # print(pointer) 
        for shard in range(self.num_shards):
            shard_indices = []
        # use subset and concatdataset to do so.
            for j, _ in enumerate(final_examples_per_shards[shard]):
                # assert len(final_examples_per_domain) == num_domains
                offset = final_examples_per_shards[shard,j]
                if offset > 0:
                    shard_indices += (indices_per_domain[j][pointer[j]:pointer[j]+offset]).tolist()
                    pointer[j] += offset
            if isinstance(dataset, WILDSSubset):
                dataset_per_shards.append(WILDSSubset(dataset.dataset, shard_indices, transform=transform))
            elif isinstance(dataset, WILDSDataset):
                dataset_per_shards.append(WILDSSubset(dataset, shard_indices, transform=transform))
            else:
                raise NotImplementedError

        assert np.array_equal(pointer, num_examples_per_domain)

        for i, dt in enumerate(dataset_per_shards):
            assert np.array_equal(np.array(np.bincount(dt.metadata_array[:,domain_field], minlength=num_domains)), np.array(final_examples_per_shards[i]))
        return dataset_per_shards


def concat_subset(subset: WILDSSubset, other: WILDSSubset):
    assert subset.dataset == other.dataset
    new_idx = subset.indices.tolist() + other.indices
    return WILDSSubset(subset.dataset, new_idx, transform=subset.transform)
