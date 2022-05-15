import os
import os.path as osp
import shutil
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree, homophily
from torch_geometric.utils import homophily, add_remaining_self_loops, to_undirected, remove_self_loops
from graph_rewiring import dirichlet_energy
from greed_params import default_params
from data import get_dataset
from data_synth_hetero import CustomDataset, Dpr2Pyg, get_edge_cat

def syn_cora_analysis(path="../data"):
  ths = ['0.00', '0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']
  df_list = []
  for targ_hom in ths:
    for rep in range(3):
      dataset = CustomDataset(root=f"{path}/syn-cora", name=f"h{str(targ_hom)}-r{str(rep+1)}", setting="gcn", seed=None)
      pyg_dataset = Dpr2Pyg(dataset)
      data = pyg_dataset.data
      num_nodes = data.num_nodes
      num_edges = data.edge_index.shape[1]
      num_classes = data.y.max() + 1
      num_features = data.num_features
      degrees = degree(data.edge_index[0], num_nodes)
      min_degree, max_degree = [degrees.min().cpu().detach().numpy(), degrees.max().cpu().detach().numpy()]
      av_degree = num_edges / num_nodes
      density = num_edges / num_nodes**2
      graph_edge_homophily = homophily(edge_index=data.edge_index, y=data.y, method='edge')
      graph_node_homophily = homophily(edge_index=data.edge_index, y=data.y, method='node')
      #label dirichlet
      de = dirichlet_energy(data.edge_index, data.edge_weight, num_nodes, data.y.unsqueeze(-1))
      edge_categories = get_edge_cat(data.edge_index, data.y, num_classes)

      row = [float(targ_hom), num_nodes, num_edges, num_classes, num_features, min_degree, max_degree, av_degree, density, graph_edge_homophily, graph_node_homophily, de, edge_categories]
      np_row = []
      for item in row:
        try:
          np_row.append(np.round(item.cpu().detach().numpy(), 4))
          # np_row.append(item.cpu().detach().numpy())
        except:
          np_row.append(np.round(item, 4))
          # np_row.append(item)
      df_list.append(np_row)
      # df_list.append(row)

  df_cols = ["target_homoph","num_nodes", "num_edges", "num_classes", "num_features", "min_degree", "max_degree", "av_degree",
             "density", "edge_homophily", "node_homophily", "label_dirichlet", "edge_categories"]

  # todo initialise spectrum distribution proportional to graph homophily
  df = pd.DataFrame(df_list, columns=df_cols)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', None)
  pd.set_option('display.max_colwidth', -1)
  df.to_csv("../ablations/syn_cora.csv")
  print(df)
  df_piv = pd.pivot_table(df.loc[:,df_cols[:-1]], values=df_cols[:-1], index='target_homoph', aggfunc=np.mean)
  df_piv = df_piv.reset_index()
  reorder = ["target_homoph","num_nodes", "num_edges" ,"num_features", "num_classes", "max_degree", "min_degree", "av_degree",
             "density", "edge_homophily", "node_homophily", "label_dirichlet"]
  df_piv = df_piv[reorder]

  reformat = ['max_degree', 'min_degree', 'av_degree', 'edge_homophily', 'node_homophily', 'label_dirichlet']
  for ref in reformat:
    # df_piv.style.format({str(ref): '{:,.2f}'})
    df_piv.loc[:, ref] = np.round(df_piv.loc[:,ref],2)
  df_piv = df_piv.rename(columns={
    "target_homoph":"homophily",
    "num_nodes":"nodes",
    "num_edges":"edges",
    "num_features":"features",
    "num_classes":"classes",
    "edge_homophily":"edge_homoph",
    "node_homophily":"node_homoph"})

  df_piv.to_csv("../ablations/syn_cora_piv.csv", index=False)
  print(df_piv)


def data_analysis(path, ds_list):
  opt = default_params()
  df_list = []
  opt['geom_gcn_splits'] = True

  for not_lcc in [True, False]:
    for undirected in [True, False]:
      for self_loops in ["orig", "non", "full"]:
        for ds in ds_list:
          opt["dataset"] = ds
          dataset = get_dataset(opt, path, not_lcc)#opt['not_lcc']) default True

          if self_loops == "full":
            dataset.data.edge_index, _ = add_remaining_self_loops(dataset.data.edge_index)
          elif self_loops == "non":
            dataset.data.edge_index, _ = remove_self_loops(dataset.data.edge_index)

          if undirected:
              dataset.data.edge_index = to_undirected(dataset.data.edge_index)

          data = dataset.data
          num_nodes = data.num_nodes
          num_edges = data.edge_index.shape[1]
          num_classes = data.y.max() + 1
          num_features = data.num_features
          degrees = degree(data.edge_index[0], num_nodes)
          min_degree, max_degree = [degrees.min().cpu().detach().numpy(), degrees.max().cpu().detach().numpy()]
          av_degree = num_edges / num_nodes
          density = num_edges / num_nodes**2
          graph_edge_homophily = homophily(edge_index=data.edge_index, y=data.y, method='edge')
          graph_node_homophily = homophily(edge_index=data.edge_index, y=data.y, method='node')
          #label dirichlet
          de = dirichlet_energy(data.edge_index, data.edge_weight, num_nodes, data.y.unsqueeze(-1))
          edge_categories = get_edge_cat(data.edge_index, data.y, num_classes)

          row = [ds, not_lcc, undirected, self_loops,
                 num_nodes, num_edges, num_classes, num_features, min_degree, max_degree, av_degree, density, graph_edge_homophily, graph_node_homophily, de, edge_categories]
          np_row = []
          for item in row[4:]:
            try:
              np_row.append(np.round(item.cpu().detach().numpy(), 4))
            except:
              np_row.append(np.round(item, 4))
          df_list.append(row[:4]+np_row)
          # df_list.append(row)

  df_cols = ["dataset", "not_lcc", "undirected", "self_loops",
             "num_nodes", "num_edges", "num_classes", "num_features", "min_degree", "max_degree", "av_degree",
             "density", "edge_homophily", "node_homophily", "label_dirichlet", "edge_categories"]

  df = pd.DataFrame(df_list, columns=df_cols)
  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', None)
  pd.set_option('display.max_colwidth', -1)
  df.to_csv("../ablations/datasets_zipped.csv")
  # print(df)
  idxs = ["dataset","not_lcc","undirected","self_loops"]
  df_piv = pd.pivot_table(df.loc[:,df_cols[:-1]], values=df_cols[:-1], index=idxs, aggfunc=np.mean)
  df_piv = df_piv.reset_index()
  reorder = ["dataset","not_lcc","undirected","self_loops",
             "num_nodes", "num_edges" ,"num_features", "num_classes", "max_degree", "min_degree", "av_degree",
             "density", "edge_homophily", "node_homophily", "label_dirichlet"]
  df_piv = df_piv[reorder]

  reformat = ['max_degree', 'min_degree', 'av_degree', 'edge_homophily', 'node_homophily', 'label_dirichlet']
  for ref in reformat:
    df_piv.loc[:, ref] = np.round(df_piv.loc[:,ref],2)
  df_piv = df_piv.rename(columns={
    "num_nodes":"nodes",
    "num_edges":"edges",
    "num_features":"features",
    "num_classes":"classes",
    "edge_homophily":"edge_homoph",
    "node_homophily":"node_homoph"})

  df_piv.to_csv("../ablations/datasets_piv_zipped.csv", index=False)
  print(df_piv)


def create_directory(dataset, old_dir, new_dir):
  raw_directory = f"{old_dir}/{dataset.lower()}/raw/"
  new_directory = f"{new_dir}/{dataset}/{dataset}/"
  new_raw_folder = new_directory + "raw/"
  try:
    # os.mkdir(new_raw_folder)
    os.makedirs(new_raw_folder)
    os.path.exists(new_raw_folder)
  except OSError:
    if os.path.exists(new_raw_folder):
      shutil.rmtree(new_raw_folder)
      shutil.copytree(raw_directory, new_raw_folder)
      print("%s exists, clearing existing data" % new_raw_folder)
    else:
      print("Creation of the directory %s failed" % new_raw_folder)
  else:
    print("Successfully created the directory %s " % new_raw_folder)

def data_zip_analysis():
  ds_list = ['Cora', 'Citeseer', 'Pubmed', 'chameleon', 'squirrel']#, 'cornell', 'texas', 'wisconsin', 'film']
  # ds_list = ['Cora']

  old_dir = "../datasets"
  new_dir = "../datasets_test"

  for ds in ds_list:
    create_directory(ds, old_dir, new_dir)

  data_analysis(new_dir, ds_list)


if __name__ == "__main__":
  # syn_cora_analysis()
  ds_list = ['Cora', 'Citeseer', 'Pubmed', 'chameleon', 'squirrel', 'cornell', 'texas', 'wisconsin', 'film']
  data_analysis(path="../data", ds_list=ds_list)
  # data_zip_analysis()

  #nb undirected graph is consistent with theory
  #don't have self loops
