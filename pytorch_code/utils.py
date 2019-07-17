#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    # 创建有向图
    graph = nx.DiGraph()
    # 遍历训练集中的每个session
    for seq in train_data:
        # 遍历session中的每个item
        for i in range(len(seq) - 1):
            # 如果当前图中没有(seq[i, seq[i+1]])的edge，则weight=1
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            # else, edge存在，则其本身的weight+1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            # 添加edge
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    # 遍历graph中的每个node, 得到normalized weight
    # 使用的edge终点的入度而非起点的出度normalize???
    for node in graph.nodes:
        sum = 0
        # 射入node的edge为(j, i), j为edge的起始node，i为结束node
        for j, i in graph.in_edges(node):
            # 累加各个edge的weight
            sum += graph.get_edge_data(j, i)['weight']
        # weight的normalization
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    # 将所有ipois补全到最大长度
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # mask
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    # shuffle
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        # data[0]为X
        inputs = data[0]
        # 输入为inputs，补全位为[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        # data[1]为Y
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph


    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        # Number of batch
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        # 等分成n_batch份
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        # 调整最后一份的length
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        # slices中储存的为多个索引list（将索引切分为各个batch），如[[0, 1, 2], [3, 4, 5], ...]
        return slices


    def get_slice(self, i):
        # inputs = inputs[i] 是一个batch
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            # n_node: Number of nodes of each session graph
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            # A node list of the session graph
            node = np.unique(u_input)
            # 用0补全session，使其达到长度最大的session的长度
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                # np.where(condition) 输出满足条件元素的坐标，以tuple的形式给出
                # 取session中的相连node，并给adjacency matrix赋值
                u = np.where(node == u_input[i])[0][0]      # 取u_input[i]对应的node标号为u
                v = np.where(node == u_input[i + 1])[0][0]  # 取u_input[i+1]对应的node标号为v
                u_A[u][v] = 1
            # Indegree adjacency matrix
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            # Outdegree adjacency matrix
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            # Concatenate
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            # 按照session的顺序，将各个item(node)对应的标号作为list存入alias_inputs
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return alias_inputs, A, items, mask, targets


















