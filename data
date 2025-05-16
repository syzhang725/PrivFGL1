import random
import re
from importlib import import_module

import numpy as np
import torch
from tensorboard.compat import tf
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected

import register

INF = np.iinfo(np.int64).max

def get_splitter(config):
    client_num = config['client_num']
    if config['data']['splitter_args']:
        kwargs = config['data']['splitter_args'][0]
    else:
        kwargs = {}

    for func in register.splitter_dict.values():
        splitter = func(config['data']['splitter'], client_num, **kwargs)
        if splitter is not None:
            return splitter

    if config['data']['splitter'] == 'louvain':
        from splitter import LouvainSplitter
        splitter = LouvainSplitter(client_num, **kwargs)

    elif config['data']['splitter'] == 'random':
        from splitter import RandomSplitter
        splitter = RandomSplitter(client_num, **kwargs)
    else:
        splitter = None

    return splitter

def get_transform(config, package):
    transform_funcs = {}
    for name in ['transform', 'target_transform', 'pre_transform']:
        if config['data'][name]:
            transform_funcs[name] = config['data'][name]

    val_transform_funcs = {}
    for name in ['val_transform', 'val_target_transform', 'val_pre_transform']:
        suf_name = name.split('val_')[1]
        if config['data'][name]:
            val_transform_funcs[suf_name] = config['data'][name]

    test_transform_funcs = {}
    for name in [
        'test_transform', 'test_target_transform', 'test_pre_transform'
    ]:
        suf_name = name.split('test_')[1]
        if config['data'][name]:
            test_transform_funcs[suf_name] = config['data'][name]

    # Transform are all None, do not import package and return dict with
    # None value
    if len(transform_funcs) and len(val_transform_funcs) and len(
        test_transform_funcs):
        return {}, {}, {}

    transforms = getattr(import_module(package), 'transforms')

    def convert(trans):
        # Recursively converting expressions to functions
        if isinstance(trans[0], str):
            if len(trans) == 1:
                trans.append({})
            transform_type, transform_args = trans
            for func in register.transform_dict.values():
                transform_func = func(transform_type, transform_args)
                if transform_func is not None:
                    return transform_func
            transform_func = getattr(transforms, transform_type)(**transform_args)
            return transform_func
        else:
            transform = [convert(x) for x in trans]
            if hasattr(transforms, 'Compose'):
                return transforms.Compose(transform)
            elif hasattr(transforms, 'Sequential'):
                return transforms.Sequential(transform)
            else:
                return transform

    # return composed transform or return list of transform
    if transform_funcs:
        for key in transform_funcs:
            transform_funcs[key] = convert(config['data'][key])

    if val_transform_funcs:
        for key in val_transform_funcs:
            val_transform_funcs[key] = convert(config['data'][key])
    else:
        val_transform_funcs = transform_funcs

    if test_transform_funcs:
        for key in test_transform_funcs:
            test_transform_funcs[key] = convert(config['data'][key])
    else:
        test_transform_funcs = transform_funcs

    return transform_funcs, val_transform_funcs, test_transform_funcs

class RegexInverseMap:
    def __init__(self, n_dic, val):
        self._items = {}
        for key, values in n_dic.items():
            for value in values:
                self._items[value] = key
        self.__val = val

    def __getitem__(self, key):
        for regex in self._items.keys():
            if re.compile(regex).match(key):
                return self._items[regex]
        return self.__val

    def __repr__(self):
        return str(self._items.items())

def load_dataset(config):
    path = config['data']['root']
    name = config['data']['type'].lower()

    splitter = get_splitter(config)
    transforms_funcs, _, _ = get_transform(config, 'torch_geometric')

    if name in ["cora", "citeseer"]:
        num_split = {
            'cora': [6, 542, INF],
            'citeseer': [332, 665, INF],
        }
        # download
        dataset = Planetoid(
            path,
            name,
            split='random',
            num_train_per_class=num_split[name][0],
            num_val=num_split[name][1],
            num_test=num_split[name][2],
            **transforms_funcs
        )

        # split
        dataset = splitter(dataset[0])

        global_dataset = Planetoid(
            path,
            name,
            split='random',
            num_train_per_class=num_split[name][0],
            num_val=num_split[name][1],
            num_test=num_split[name][2],
            **transforms_funcs
        )
    else:
        raise ValueError(f'No dataset named: {name}!')

    dataset = [ds for ds in dataset]  # convert the dataset to a list
    client_num = min(len(dataset), config['client_num']) if config['client_num'] > 0 else len(dataset)
    config['client_num'] = client_num

    # get local dataset
    data_dict = dict()
    for client_idx in range(1, len(dataset) + 1):
        local_data = dataset[client_idx - 1]

        # To undirected and add self-loop
        local_data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(local_data.edge_index)[0]),
            num_nodes=local_data.x.shape[0]
        )[0]
        data_dict[client_idx] = {
            'data': local_data,
            'train': [local_data],
            'val': [local_data],
            'test': [local_data]
        }
    # Keep ML split consistent with local graphs
    if global_dataset is not None:
        global_graph = global_dataset[0]
        train_mask = torch.zeros_like(global_graph.train_mask)
        val_mask = torch.zeros_like(global_graph.val_mask)
        test_mask = torch.zeros_like(global_graph.test_mask)

        for client_sampler in data_dict.values():
            if isinstance(client_sampler, Data):
                client_subgraph = client_sampler
            else:
                client_subgraph = client_sampler['data']
            train_mask[client_subgraph.index_orig[client_subgraph.train_mask]] = True
            val_mask[client_subgraph.index_orig[client_subgraph.val_mask]] = True
            test_mask[client_subgraph.index_orig[client_subgraph.test_mask]] = True
        global_graph.train_mask = train_mask
        global_graph.val_mask = val_mask
        global_graph.test_mask = test_mask

        data_dict[0] = {
            'data': global_graph,
            'train': [global_graph],
            'val': [global_graph],
            'test': [global_graph]
        }

    return data_dict
