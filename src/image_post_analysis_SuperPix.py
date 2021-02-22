import argparse
import torch
import numpy as np
import os
from GNN_image_pixel import GNN_image_pixel
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch_geometric.utils import to_dense_adj
import pandas as pd
from data_image import load_pixel_data
import shutil
from MNIST_SuperPix import load_SuperPixel_data, transform_objects, get_centroid_coords_array
import skimage
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage import segmentation
import networkx as nx
from torch_geometric.utils.convert import to_networkx


def UnNormalizeCIFAR(data):
  #normalises each image channel to range [0,1] from [-1, 1]
  # return (data - torch.amin(data,dim=(0,1))) / (torch.amax(data,dim=(0,1)) - torch.amin(data,dim=(0,1)))
  return data * 0.5 + 0.5

def check_folder(savefolder):
  try:
    os.mkdir(savefolder)
  except OSError:
    if os.path.exists(savefolder):
      shutil.rmtree(savefolder)
      os.mkdir(savefolder)
      print("%s exists, clearing existing images" % savefolder)
    else:
      print("Creation of the directory %s failed" % savefolder)
  else:
    print("Successfully created the directory %s " % savefolder)


# def get_paths(modelpath, model_key, opt, Tmultiple, partitions, batch_num):
#   #load data and model
#   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#   SuperPixelData = load_SuperPixel_data(opt)
#   loader = DataLoader(SuperPixelData, batch_size=opt['batch_size'], shuffle=False)  # True)
#   for batch_idx, batch in enumerate(loader):
#     if batch_idx == batch_num:
#       break
#   batch.to(device)
#   edge_index_gpu = batch.edge_index
#   edge_attr_gpu = batch.edge_attr
#   if edge_index_gpu is not None: edge_index_gpu.to(device)
#   if edge_attr_gpu is not None: edge_index_gpu.to(device)
#   opt['time'] = opt['time'] / partitions
#   model = GNN_image_pixel(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
#                     batch.edge_attr, device).to(device)
#
#   model.load_state_dict(torch.load(f"{modelpath}.pt", map_location=device))
#   model.to(device)
#   model.eval()
#   ###do forward pass
#   for batch_idx, batch in enumerate(loader):
#     if batch_idx == batch_num:
#       paths, atts = model.forward_plot_SuperPix(batch.x.to(model.device), Tmultiple * partitions)
#       pix_labels = batch.y
#       train_mask = batch.train_mask
#       labels = batch.target
#       break
#
#   paths_nograd = paths.cpu().detach()
#   atts_nograd = atts #.cpu().detach()
#   pix_labels_nograd = pix_labels.cpu().detach()
#   labels_nograd = labels.cpu().detach()
#   train_mask_nograd = train_mask.cpu().detach()
#
#   return paths_nograd, atts_nograd, pix_labels_nograd, labels_nograd, train_mask_nograd


def build_batches(model_keys, model_epochs, samples, Tmultiple, partitions, batch_num, times):
  directory = f"../SuperPix/"
  df = pd.read_csv(f'{directory}models.csv')
  for i, model_key in enumerate(model_keys):
    for filename in os.listdir(directory):
      if filename.startswith(model_key):
        path = os.path.join(directory, filename)
        print(path)
        break
    [_, _, data_name, blck, fct] = path.split("_")
    modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
    modelpath = f"{modelfolder}/model_{model_key}_epoch{model_epochs[i]}"
    optdf = df[df.model_key == model_key]
    intcols = ['num_class','im_chan','im_height','im_width','num_nodes']
    optdf[intcols].astype(int)
    opt = optdf.to_dict('records')[0]

    # paths, atts, pix_labels, labels, train_mask = get_paths(modelpath, model_key, opt, Tmultiple, partitions, batch_num)
    # batch_vis_mask(train_mask, paths, pix_labels, labels, opt, pic_folder=modelfolder, samples=samples)
    # batch_vis_labels(train_mask, pix_labels, labels, opt, pic_folder=modelfolder, samples=samples)
    # batch_vis_masklabels(train_mask, pix_labels, labels, opt, pic_folder=modelfolder, samples=samples)
    # batch_image(paths, labels, time=0, opt=opt, pic_folder=modelfolder, samples=samples)
    # batch_image(paths, labels, time=5, opt=opt, pic_folder=modelfolder, samples=samples)
    # batch_image(paths, labels, time=10, opt=opt, pic_folder=modelfolder, samples=samples)
    # batch_animation(paths, labels, Tmultiple * partitions, fps=2, opt=opt, pic_folder=modelfolder, samples=samples)
    # batch_pixel_intensity(paths, labels, opt, pic_folder=modelfolder, samples=samples)
    view_SuperPix_Att(modelfolder, modelpath, model_key, opt, Tmultiple, partitions, batch_num, samples, times)


def view_SuperPix_Att(modelfolder, modelpath, model_key, opt, Tmultiple, partitions, batch_num, samples, times):
  #load data and model   ###do forward pass
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  SuperPixelData = load_SuperPixel_data(opt)
  loader = DataLoader(SuperPixelData, batch_size=1, shuffle=True)#False)  # True)
  opt['time'] = opt['time'] / partitions
  for batch_idx, batch in enumerate(loader):
    if batch_idx == samples:
      break
    batch.to(device)
    edge_index_gpu = batch.edge_index
    edge_attr_gpu = batch.edge_attr
    if edge_index_gpu is not None: edge_index_gpu.to(device)
    if edge_attr_gpu is not None: edge_index_gpu.to(device)

    model = GNN_image_pixel(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                            edge_attr_gpu, device).to(device)

    model.load_state_dict(torch.load(f"{modelpath}.pt", map_location=device))
    model.to(device)
    model.eval()
    paths, atts = model.forward_plot_SuperPix(batch.x.to(model.device), Tmultiple * partitions)
    SuperPixItem = batch

    im_height = opt['im_height']
    im_width = opt['im_width']
    SF = 448  # 56 #480  # 480    #RESIZING needed as mark_boundaries marks 1 pixel wide either side of boundary
    heightSF = SF / im_height
    widthSF = SF / im_width

    pixel_values = SuperPixItem.orig_image.detach().numpy()
    pixel_labels = SuperPixItem.pixel_labels.view(im_height, im_width)
    x = SuperPixItem.x
    broadcast_pixels = x[pixel_labels].squeeze().numpy()
    centroids = SuperPixItem.centroids[0]
    num_centroids = torch.max(pixel_labels) + 1
    r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                                    pixel_values, pixel_labels.numpy(), centroids)
    r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)
    r_pixel_labels = r_pixel_labels.astype(np.int)
    out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))
    NXgraph = to_networkx(SuperPixItem)

    #####PLOT T=0 before diffusion
    fig, ax = plt.subplots()    # fig, (ax, cbar_ax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [1, 0.05]})
    ax.axis('off')  # , cbar_ax.axis('off')
    ax.imshow(out)
    for i in range(num_centroids):
      # ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
      label = SuperPixItem.y[i].item()
      # ax.annotate(f"{i}:{label}", (r_x_coords[i], r_y_coords[i]), c="red")
      ax.annotate(label, (r_x_coords[i], r_y_coords[i]), c="red")

    ax.scatter(x=r_x_coords, y=r_y_coords)
    # edge_weights = atts[time]
    time = 0
    x = paths[:, time, :].detach().numpy()
    edge_weights = atts[time].detach().numpy() #* (x[SuperPixItem.edge_index[0, :]].squeeze()
                               #  + x[SuperPixItem.edge_index[1, :]].squeeze())
    weight_max = 5
    edge_weights = ((edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())) * (
              weight_max - 1) + 1
    nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color=list(edge_weights), #"lime",
            node_color=x, cmap=plt.get_cmap('Spectral'), width=list(edge_weights))
    # fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('Spectral')),
    #              cax=cbar_ax, orientation="vertical")
    plt.title(f"t={time} Attention, Ground Truth: {SuperPixItem.target.item()}")
    check_folder(f"{modelfolder}/sample_{batch_idx}")
    plt.savefig(f"{modelfolder}/sample_{batch_idx}/initial_{time}.pdf", format="pdf")
    plt.show()


    ######PLOT BROADCAST DIFFUSION
    for time in times:
      x = paths[:, time, :].detach().numpy()
      broadcast_pixels = x[pixel_labels].squeeze()
      r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                                      broadcast_pixels, pixel_labels.numpy(), centroids)
      r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)
      r_pixel_labels = r_pixel_labels.astype(np.int)
      out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))

      fig, ax = plt.subplots()
      ax.axis('off')
      ax.imshow(out)
      for i in range(num_centroids):
        # ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
        prediction = round(x[i].item(),2)
        # ax.annotate(f"{i}:{prediction}", (r_x_coords[i], r_y_coords[i]), c="red")
        ax.annotate(prediction, (r_x_coords[i], r_y_coords[i]), c="red")
      ax.scatter(x=r_x_coords, y=r_y_coords)
      edge_weights = atts[time].detach().numpy() #* (x[SuperPixItem.edge_index[0,:]].squeeze())
                                   # + x[SuperPixItem.edge_index[1,:]].squeeze())
      edge_weights = ((edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())) * (weight_max-1) + 1

      nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color=list(edge_weights), #"lime",
              node_color=x, cmap=plt.get_cmap('Spectral'), width=list(edge_weights))

      plt.title(f"t={time} Attention, Ground Truth: {SuperPixItem.target.item()}")
      plt.savefig(f"{modelfolder}/sample_{batch_idx}/diffused_{time}.pdf", format="pdf")
      plt.show()

      #save down attention matrices
      dense_att = to_dense_adj(edge_index=SuperPixItem.edge_index, edge_attr=atts[time].detach(),
                              max_num_nodes=num_centroids)[0,:,:]
      square_att = dense_att.view(num_centroids, num_centroids)
      x_np = square_att.detach().numpy()
      x_df = pd.DataFrame(x_np)
      x_df.to_csv(f"{modelfolder}/sample_{batch_idx}/att_{time}.csv")

    # #plot attention paths
    # atts = torch.stack(atts,dim=1).T
    # fig = plt.figure()
    # plt.tight_layout()
    # plt.plot(atts)
    # plt.title("Attentions Evolution")
    # plt.savefig(f"{modelfolder}/atts_evol.pdf", format="pdf")
    # plt.show()

    ####Plot max/min pixel intensity
    fig = plt.figure()
    plt.tight_layout()
    if opt['im_dataset'] == 'MNIST':
      A = paths[:, :, 0].detach().cpu()
      plt.plot(torch.max(A,dim=0)[0], color='red')
      plt.plot(torch.min(A,dim=0)[0], color='green')
      plt.plot(torch.mean(A,dim=0), color='blue')
    plt.title("Max/Min, Ground Truth: {}".format(SuperPixItem.target.item()))
    plt.savefig(f"{modelfolder}/sample_{batch_idx}/max_min.pdf", format="pdf")
    plt.show()

    ###### CREATE ANIMATION
    #draw initial graph
    time = 0
    x = paths[:, time, :].detach().numpy()
    broadcast_pixels = x[pixel_labels].squeeze()
    r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                                    broadcast_pixels, pixel_labels.numpy(), centroids)
    r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)
    r_pixel_labels = r_pixel_labels.astype(np.int)
    out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(out)
    for i in range(num_centroids):
      ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
    ax.scatter(x=r_x_coords, y=r_y_coords)
    edge_weights = atts[time].detach().numpy() #* (x[SuperPixItem.edge_index[0,:]].squeeze()
                               #  + x[SuperPixItem.edge_index[1,:]].squeeze())
    edge_weights = ((edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())) * (weight_max-1) + 1

    nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color=list(edge_weights), #"lime",
            node_color=x, cmap=plt.get_cmap('Spectral'), width=list(edge_weights))
    plt.title(f"t={time} Attention, Ground Truth: {SuperPixItem.target.item()}")

    # loop through data and update plot
    def update(ii):
      plt.tight_layout()
      x = paths[:, ii, :].detach().numpy()
      broadcast_pixels = x[pixel_labels].squeeze()
      r_pixel_values, r_pixel_labels, r_centroids = transform_objects(im_height, im_width, heightSF, widthSF,
                                                                      broadcast_pixels, pixel_labels.numpy(), centroids)
      r_y_coords, r_x_coords = get_centroid_coords_array(num_centroids, r_centroids)
      r_pixel_labels = r_pixel_labels.astype(np.int)
      out = segmentation.mark_boundaries(r_pixel_values, r_pixel_labels, (1, 0, 0))
      ax.imshow(out)
      for i in range(num_centroids):
        ax.annotate(i, (r_x_coords[i], r_y_coords[i]), c="red")
      ax.scatter(x=r_x_coords, y=r_y_coords)
      edge_weights = atts[ii].detach().numpy() #* (x[SuperPixItem.edge_index[0,:]].squeeze()
                               #    + x[SuperPixItem.edge_index[1,:]].squeeze())
      edge_weights = ((edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())) * (weight_max-1) + 1

      nx.draw(NXgraph, r_centroids, ax=ax, node_size=300 / 4, edge_color=list(edge_weights), #"lime",
              node_color=x, cmap=plt.get_cmap('Spectral'), width=list(edge_weights))
      plt.title(f"t={ii} Attention, Ground Truth: {SuperPixItem.target.item()}")

    fig = plt.gcf()
    frames = 10
    fps = 1.5
    animation = FuncAnimation(fig, func=update, frames=frames)
    animation.save(f"{modelfolder}/sample_{batch_idx}/animation.gif", fps=fps)#, writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=fps)

def attention_evolution(model_keys, attention_epochs, samples, Tmultiple, partitions, batch_num, times):
    directory = f"../SuperPix/"
    df = pd.read_csv(f'{directory}models.csv')
    for i, model_key in enumerate(model_keys):
      for filename in os.listdir(directory):
        if filename.startswith(model_key):
          path = os.path.join(directory, filename)
          print(path)
          break
      [_, _, data_name, blck, fct] = path.split("_")
      modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
      optdf = df[df.model_key == model_key]
      intcols = ['num_class', 'im_chan', 'im_height', 'im_width', 'num_nodes']
      optdf[intcols].astype(int)
      opt = optdf.to_dict('records')[0]
      opt['time'] = opt['time'] / partitions
      # load data and model   ###do forward pass
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      SuperPixelData = load_SuperPixel_data(opt)
      loader = DataLoader(SuperPixelData, batch_size=1, shuffle=False)  # True)
      for batch_idx, batch in enumerate(loader):
        if batch_idx == samples:
          break
        batch.to(device)
        edge_index_gpu = batch.edge_index
        edge_attr_gpu = batch.edge_attr
        if edge_index_gpu is not None: edge_index_gpu.to(device)
        if edge_attr_gpu is not None: edge_index_gpu.to(device)
        model = GNN_image_pixel(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                                edge_attr_gpu, device).to(device)

        for att_epoch in attention_epochs:
          modelpath = f"{modelfolder}/model_{model_key}_epoch{att_epoch}"
          model.load_state_dict(torch.load(f"{modelpath}.pt", map_location=device))
          model.to(device)
          model.eval()
          paths, atts = model.forward_plot_SuperPix(batch.x.to(model.device), Tmultiple * partitions)

          # plot attention paths
          atts = torch.stack(atts, dim=1).detach().T
          fig = plt.figure()
          plt.tight_layout()
          plt.plot(atts)
          plt.title("Attentions Evolution")
          plt.savefig(f"{modelfolder}/sample_{batch_idx}/atts_evol_epoch{att_epoch}.pdf", format="pdf")
          # plt.show()

@torch.no_grad()
def model_comparison(model_keys, model_epochs, times, image_folder, samples, Tmultiple, partitions, batch_num):
    directory = f"../SuperPix/"
    df = pd.read_csv(f'{directory}models.csv')
    savefolder = f"../SuperPix/images/{image_folder}"
    check_folder(savefolder)

    for sample in range(samples):
      images_A = []
      images_B = []
      images_C = []
      label_list = []
      datasets = []
      for i, model_key in enumerate(model_keys):
        for filename in os.listdir(directory):
          if filename.startswith(model_key):
            path = os.path.join(directory, filename)
            print(path)
            break
        [_, _, data_name, blck, fct] = path.split("_")
        modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
        modelpath = f"{modelfolder}/model_{model_key}_epoch{model_epochs[i]}"
        optdf = df[df.model_key == model_key]
        intcols = ['num_class', 'im_chan', 'im_height', 'im_width', 'num_nodes']
        optdf[intcols].astype(int)
        opt = optdf.to_dict('records')[0]


        A = paths[sample, times[0], :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
        B = paths[sample, times[1], :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
        C = paths[sample, times[2], :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
        images_A.append(A)
        images_B.append(B)
        images_C.append(C)
        label_list.append(f"{opt['block']}\n{opt['function']}")
        datasets.append(opt['im_dataset'])
      images = [images_A, images_B, images_C]

      label_list.append(f"{opt['block']}\n{opt['function']}")
      new_label_dict = {f"constant\nlaplacian" : f"constant\nlaplacian",
                     f"attention\nlaplacian": f"GRAND-l",
                     f"constant\ntransformer": f"GRAND-nl"}
      new_label_list = [new_label_dict[label] for label in label_list]

      plot_times = [f"t={int(opt['time'] * time)}" for time in times]

      fig, axs = plt.subplots(3,3, figsize=(9, 6), sharex=True, sharey=True)
      fig.suptitle(f"{opt['im_dataset']} SuperPixel Diffusion")

      for i in range(3):
        for j in range(3):
                axs[i, j].imshow(images[j][i], cmap='gray', interpolation='none')
                axs[i,j].set_yticks([])
                axs[i,j].set_xticks([])
        plt.subplots_adjust(wspace=-0.2, hspace=0)
        for ax, t in zip(axs[0], plot_times):
            ax.set_title(t, size=18)
        pad = 2
        for ax, row in zip(axs[:,0], new_label_list):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
        plt.savefig(f"{savefolder}/sample_{sample}.pdf",format="pdf")


if __name__ == '__main__':
  Tmultiple = 1
  partitions = 10  #partitions of each T = 1
  batch_num = 0 #1#2
  samples = 6
  times = [0,1*Tmultiple,5*Tmultiple,10*Tmultiple]
  # directory = f"../pixels/"
  # df = pd.read_csv(f'{directory}models.csv')
  # model_keys = df['model_key'].to_list()

  # model_keys = ['20210212_101642','20210217_101830']
  # model_keys = ['20210217_101830']#20210217_130606']
  model_keys = ['20210218_132704']#['20210218_104802','20210218_110011']

  model_epochs = [63]#[20]#,7]#, 4] #, 63, 63] #, 15, 15]
  build_batches(model_keys, model_epochs, samples, Tmultiple, partitions, batch_num, times)

  # model_keys = ['20210217_101830']#20210217_130606'] #['20210217_101830']
  # attention_epochs = [7,15,63]
  attention_epochs = [0,1,2,4,7,31,63]
  attention_evolution(model_keys, attention_epochs, samples, Tmultiple, partitions, batch_num, times)

  # model_comparison
  model_keys = ['20210212_101642','20210221_173048','20210218_132704']
  model_epochs = [63,63,63]
  times = [1, 3, 5]
  image_folder = 'MNIST_Superpix'
  model_comparison(model_keys, model_epochs, times, image_folder, samples, Tmultiple, partitions, batch_num)
