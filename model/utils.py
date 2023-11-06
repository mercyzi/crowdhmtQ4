import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import torch
from transformers import BertModel
from transformers import BertTokenizer
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.data as data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
import random
from scipy.ndimage import gaussian_filter1d

#加载预训练模型
pretrained = BertModel.from_pretrained('/home/zxh/clinical/src/model/chinese_wwm_pytorch')
tokenizer = BertTokenizer.from_pretrained('/home/zxh/clinical/src/model/chinese_wwm_pytorch')
#需要移动到cuda上
pretrained.to(device)
#不训练,不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)

import json

def load_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data

def token_encode(input):
    return tokenizer.encode(text = input, truncation=True,
                             padding='max_length',   # 少于max_length时就padding
                            add_special_tokens=True,
                            max_length=100,
                            return_tensors=None,  # None表示不指定数据类型，默认返回list)
                            )

def clean_speaker(speaker):
    if '对' in speaker:
        return speaker[1:-1]
    else:
        return speaker
    
def to_tensor_(array, dtype=None):
    if dtype is None:
        return torch.tensor(array, device=device)
    else:
        return torch.tensor(array, dtype=dtype, device=device)

def preprocess(samples):
    new_samples = []
    for sample in samples:
        new_sample = {}
        text = to_tensor_(token_encode(sample['Topic']), dtype=torch.long).unsqueeze(0)
        src_key_padding_mask = text.eq(0).unsqueeze(0).contiguous()
        
        new_sample['Topic'] = pretrained(text, src_key_padding_mask).last_hidden_state[:, 0].cpu().numpy().tolist()
        new_sample['Debate'] = []
        for diag in sample['Debate']:
            speaker = clean_speaker(diag['speaker'])
            text = to_tensor_(token_encode(diag['content']), dtype=torch.long).unsqueeze(0)
            src_key_padding_mask = text.eq(0).unsqueeze(0).contiguous()
            content = pretrained(text, src_key_padding_mask).last_hidden_state[:, 0].cpu().numpy().tolist()
            object = clean_speaker(diag['object'])
            new_sample['Debate'].append({'speaker': speaker,
                                         'content': content,
                                         'object': object})
        
        new_samples.append(new_sample)   

    return new_samples

class Sample_nodes():
    def __init__(self, sample, node_start):
        self.name2node = {}
        self.cur_node = node_start
        self.update_inf(sample, node_start)

    def update_inf(self, sample, node_start):
        node = node_start
        for diag in sample['Debate']:
            if diag['speaker'] not in self.name2node.keys():
                self.name2node[diag['speaker']] = node
                node = node + 1
            if diag['object'] != '':
                if diag['object'] not in self.name2node.keys():
                    self.name2node[diag['object']] = node
                    node = node + 1
        self.cur_node = node

def make_labels(node_min, node_max, node_x, node_label):
    edge_lables = [0, 1]
    edge_lable_x = []
    edge_lable_y = []
    for idx in range(node_min, node_max):
        if idx != node_x and idx!= node_label:
            edge_lable_x.append(node_x)
            edge_lable_y.append(idx)
    edge_lable_x.append(node_x)
    edge_lable_y.append(node_label)
    edge_lables[0] = edge_lable_x
    edge_lables[1] = edge_lable_y
    return edge_lables


def create_graph(ds_list: list):
    
    graphs = []
    for sample in ds_list:
        node = 0
        x= []
        edge_index = []
        edge_attr = []
        edge_label_index = [[],[]]
        ground_label = []
        edge_label_attr = []
        topic = []
        node_inf = Sample_nodes(sample, node)
        for diag in sample['Debate']:
            if diag is not sample['Debate'][-1]:
                speaker_node = node_inf.name2node[diag['speaker']]
                object_node = node_inf.name2node[diag['object']]
                edge_index.append([speaker_node, object_node])
                edge_attr.append(diag['content'])
        edge_label_index = make_labels(node, node_inf.cur_node, node_inf.name2node[sample['Debate'][-1]['speaker']], node_inf.name2node[sample['Debate'][-1]['object']])    
        for i in range(len(edge_label_index[0])-1):
            ground_label.append(0)
            edge_label_attr.append(sample['Debate'][-1]['content'][0])
            topic.append(sample['Topic'][0])
        ground_label.append(1)
        edge_label_attr.append(sample['Debate'][-1]['content'][0])
        topic.append(sample['Topic'][0])
        for i in range(node_inf.cur_node):
            number = torch.LongTensor([i])
            x.append(number)
        x = torch.LongTensor(x)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attr).squeeze(1)
        g = data.Data(edge_index=edge_index, x=x, edge_attr=edge_attr)
        g.edge_label_index = torch.LongTensor(edge_label_index)
        g.ground_label = torch.FloatTensor(ground_label)
        g.edge_label_attr = torch.FloatTensor(edge_label_attr)
        g.topic = torch.FloatTensor(topic)
        
        graphs.append(g)
    
    return graphs

def sample_balance_mask(batch):
    indices = torch.nonzero(batch.ground_label == 1).squeeze()  
    indices_s = indices + 1
    indices_s = torch.cat((torch.tensor([0]).to(device), indices_s[:-1]))
    sampled_batch_mask = torch.zeros(batch.ground_label.size()[0]).bool().to(device)
    for start, end in zip(indices_s, indices):
        sampled_batch_mask[end] = True
        sampled_batch_mask[random.randrange(start, end)] = True
    
    return sampled_batch_mask

# def get_shuffle_train(batch_size, data):

#     indices = torch.nonzero(data.train_ground_label == 1).squeeze()  
#     indices_s = indices + 1
#     indices_s = torch.cat((torch.tensor([0]).to(device), indices_s[:-1]))
#     # 获取原始 tensor 的长度
#     tensor_length = indices.size(0)
#     sampled_batch_label = [[],[]]
#     sampled_batch_mask = torch.zeros(data.train_mask.size()[0]).bool().to(device)
#     sampled_batch_ground_label = []
#     sampled_batch_edge_label_attr = []
#     # 生成随机的索引，此处示例抽取 5 个值
#     random_indx = torch.randperm(tensor_length)[:batch_size]
#     for i,j in zip(indices_s[random_indx], indices[random_indx]):
#         for indx in range(i,j+1):
#             sampled_batch_label[0].append(data.train_label[0][indx])
#             sampled_batch_label[1].append(data.train_label[1][indx])
#             sampled_batch_mask[indx] = True
#             sampled_batch_ground_label.append(data.train_ground_label[indx])
#             sampled_batch_edge_label_attr.append(data.edge_label_attr[indx])
#     sampled_batch_label = torch.tensor(sampled_batch_label).to(device)

#     sampled_batch_mask = torch.tensor(sampled_batch_mask).to(device)

#     sampled_batch_ground_label = torch.tensor(sampled_batch_ground_label).to(device)

#     sampled_batch_edge_label_attr = torch.stack(sampled_batch_edge_label_attr).to(device)
    

#     return sampled_batch_label, sampled_batch_mask, sampled_batch_ground_label, sampled_batch_edge_label_attr
    
