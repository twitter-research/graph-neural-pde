import torch
import sys
import networkx as nx
from dgl import DGLGraph
import numpy as np
from dgl.data import load_data
import dgl
import os
import scipy.sparse as sp
import pickle
import collections
import numpy


def load_x(filename):
    if sys.version_info > (3, 0):
        return pickle.load(open(filename, 'rb'), encoding='latin1')
    else:
        return numpy.load(filename)


def ReadMixhopDataset(mixhop_dataset_path, device):
    edge_lists = load_x(mixhop_dataset_path + '.graph')

    allx = np.load(mixhop_dataset_path + '.allx')
    ally = numpy.array(numpy.load(mixhop_dataset_path + '.ally'), dtype='float32')

    num_nodes = len(edge_lists)

    # Will be used to construct (sparse) adjacency matrix.
    edge_sets = collections.defaultdict(set)
    for node, neighbors in edge_lists.items():
        edge_sets[node].add(node)  # Add self-connections
        for n in neighbors:
            edge_sets[node].add(n)
            edge_sets[n].add(node)  # Assume undirected.

    # Now, build adjacency list.
    adj_indices = []
    for node, neighbors in edge_sets.items():
        for n in neighbors:
            adj_indices.append((node, n))

    adj_indices = numpy.array(adj_indices, dtype='int32')

    graph = dgl.graph((adj_indices[:, 0], adj_indices[:, 1]), num_nodes=num_nodes)
    features = torch.Tensor(allx)
    num_labels = ally.shape[1]
    labels = torch.LongTensor(ally.argmax(1))

    train_mask, val_mask, test_mask = [torch.BoolTensor(num_nodes).fill_(False) for _ in range(3)]
    train_mask[:num_nodes // 3] = True
    val_mask[num_nodes // 3: 2 * num_nodes // 3] = True
    test_mask[2 * num_nodes // 3:] = True

    return (graph.to(device),
            num_labels,
            features.to(device),
            labels.to(device),
            train_mask.to(device),
            val_mask.to(device),
            test_mask.to(device))


def load_dataset(args, device):
    self_loop = args.self_loop
    try:
        dataset = load_data(args)
        features = torch.FloatTensor(dataset.features)
        labels = torch.LongTensor(dataset.labels)

        # graph preprocess
        graph = dataset.graph
        # add self loop
        if self_loop:
            graph.remove_edges_from(nx.selfloop_edges(graph))
            graph.add_edges_from(zip(graph.nodes(), graph.nodes()))
        graph = DGLGraph(graph)

        train_mask = torch.BoolTensor(dataset.train_mask)
        val_mask = torch.BoolTensor(dataset.val_mask)
        test_mask = torch.BoolTensor(dataset.test_mask)

        num_labels = dataset.num_labels

    except ValueError:  # The dataset is not present in the DGL library
        if 'ogbn' in args.dataset:
            dataset = DglNodePropPredDataset(name=args.dataset)
            graph = dgl.to_simple(dataset[0][0])
            labels = dataset[0][1].view(-1)
            num_labels = dataset.num_classes
            split_idx = dataset.get_idx_split()
            n_nodes = graph.number_of_nodes()
            features = graph.ndata['feat']

            def to_mask(indx):
                mask = torch.zeros(n_nodes, dtype=torch.bool)
                mask[indx] = 1
                return mask

            if self_loop:
                graph.remove_edges_from(nx.selfloop_edges(graph))
                graph.add_edges_from(zip(graph.nodes(), graph.nodes()))

            train_mask, val_mask, test_mask = map(to_mask, (split_idx['train'], split_idx['valid'], split_idx['test']))
        else:
            ###The following code is adapted from the GeomGCN code base https://github.com/graphdml-uiuc-jlu/geom-gcn
            assert not (args.original_split), 'no original split available for dataset ' + args.dataset

            def preprocess_features(features):
                """Row-normalize feature matrix and convert to tuple representation"""
                rowsum = np.array(features.sum(1))
                r_inv = np.power(rowsum, -1).flatten()
                r_inv[np.isinf(r_inv)] = 0.
                r_mat_inv = sp.diags(r_inv)
                features = r_mat_inv.dot(features)
                return features

            graph_adjacency_list_file_path = os.path.join(args.datasets_path, args.dataset, 'out1_graph_edges.txt')
            graph_node_features_and_labels_file_path = os.path.join(args.datasets_path, args.dataset,
                                                                    f'out1_node_feature_label.txt')
            graph_labels_dict = {}
            G = nx.DiGraph()
            graph_node_features_dict = {}

            if args.dataset == 'film':
                with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                    graph_node_features_and_labels_file.readline()
                    for line in graph_node_features_and_labels_file:
                        line = line.rstrip().split('\t')
                        assert (len(line) == 3)
                        assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                        feature_blank = np.zeros(932, dtype=np.uint8)
                        feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                        graph_node_features_dict[int(line[0])] = feature_blank
                        graph_labels_dict[int(line[0])] = int(line[2])
            else:
                with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                    graph_node_features_and_labels_file.readline()
                    for line in graph_node_features_and_labels_file:
                        line = line.rstrip().split('\t')
                        assert (len(line) == 3)
                        assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                        graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                        graph_labels_dict[int(line[0])] = int(line[2])

            with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
                graph_adjacency_list_file.readline()
                for line in graph_adjacency_list_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 2)
                    if int(line[0]) not in G:
                        G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                   label=graph_labels_dict[int(line[0])])
                    if int(line[1]) not in G:
                        G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                   label=graph_labels_dict[int(line[1])])
                    G.add_edge(int(line[0]), int(line[1]))

            adj = nx.adjacency_matrix(G, sorted(G.nodes()))
            features = np.array(
                [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
            labels = np.array(
                [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

            features = torch.FloatTensor(preprocess_features(features))
            graph = DGLGraph(adj + sp.eye(adj.shape[0]))
            labels = torch.LongTensor(labels)

            num_labels = len(np.unique(labels))
            # print('num labels ',np.unique(labels,return_counts = True))

    if args.custom_split_file:
        with np.load(args.custom_split_file) as splits_file:
            train_mask = torch.BoolTensor(splits_file['train_mask'])
            val_mask = torch.BoolTensor(splits_file['val_mask'])
            test_mask = torch.BoolTensor(splits_file['test_mask'])
    elif not (args.original_split):
        n_nodes = features.size(0)
        split_indices = np.arange(n_nodes)

        reseed = np.random.randint(2 ** 31)
        np.random.seed(args.split_seed)
        np.random.shuffle(split_indices)
        label_indices = [(labels.numpy() == i).nonzero()[0] for i in range(num_labels)]
        for i in range(num_labels):
            np.random.shuffle(label_indices[i])

        np.random.seed(reseed)
        train_mask = torch.BoolTensor(n_nodes).fill_(0)
        val_mask = torch.BoolTensor(n_nodes).fill_(0)
        test_mask = torch.BoolTensor(n_nodes).fill_(0)

        if args.small_train_split:
            train_point_fraction = 0.1
            val_point_fraction = 0.2
        else:
            train_point_fraction = 0.6
            val_point_fraction = 0.8

        if args.homogeneous_split:
            for i in range(num_labels):
                indices = label_indices[i]
                train_point = int(train_point_fraction * len(indices))
                val_point = int(val_point_fraction * len(indices))

                train_mask[indices[:train_point]] = 1
                val_mask[indices[train_point:val_point]] = 1
                test_mask[indices[val_point:]] = 1

        else:
            train_point = int(train_point_fraction * n_nodes)
            val_point = int(val_point_fraction * n_nodes)
            train_mask[split_indices[:train_point]] = 1
            val_mask[split_indices[train_point:val_point]] = 1
            test_mask[split_indices[val_point:]] = 1

    return (graph.to(device),
            num_labels,
            features.to(device),
            labels.to(device),
            train_mask.to(device),
            val_mask.to(device),
            test_mask.to(device))
