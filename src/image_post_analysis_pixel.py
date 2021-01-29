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
from data_image import load_data
from image_opt import get_image_opt
import shutil
from collections import OrderedDict


def UnNormalizeCIFAR(data):
  #normalises each image channel to range [0,1] from [-1, 1]
  # return (data - torch.amin(data,dim=(0,1))) / (torch.amax(data,dim=(0,1)) - torch.amin(data,dim=(0,1)))
  return data * 0.5 + 0.5

@torch.no_grad()
def plot_image_T(paths, labels, T_idx, opt, height=2, width=3):
  fig = plt.figure()
  for i in range(height*width):
    # t == 0
    plt.subplot(2*height, width, i + 1)
    plt.tight_layout()
    plt.axis('off')
    A = paths[i,0,:].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title("t=0 Ground Truth: {}".format(labels[i].item()))

    #t == T
    plt.subplot(2*height, width, height*width + i + 1)
    plt.tight_layout()
    plt.axis('off')
    A = paths[i, T_idx, :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
    A = A.view(opt['im_height'], opt['im_width'], opt['im_chan'])
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title("t=T Ground Truth: {}".format(labels[i].item()))
  return fig


@torch.no_grad()
def create_animation_old(paths, labels, opt, height, width, frames):
  # draw graph initial graph
  fig = plt.figure()
  for i in range(height * width):
    plt.subplot(height, width, i + 1)
    plt.tight_layout()
    A = paths[i, 0, :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation='none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A)
    plt.title("t=0 Ground Truth: {}".format(labels[i].item()))
    plt.axis('off')

  # loop through data and update plot
  def update(ii):
    for i in range(height * width):
      plt.subplot(height, width, i + 1)
      plt.tight_layout()
      A = paths[i, ii, :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
      if opt['im_dataset'] == 'MNIST':
        plt.imshow(A, cmap='gray', interpolation='none')
      elif opt['im_dataset'] == 'CIFAR':
        A = UnNormalizeCIFAR(A)
        plt.imshow(A)
      plt.title("t={} Ground Truth: {}".format(ii, labels[i].item()))
      plt.axis('off')
  fig = plt.gcf()
  animation = FuncAnimation(fig, func=update, frames=frames)#, blit=True)
  return animation


@torch.no_grad()
def create_pixel_intensity_old(paths, labels, opt, height, width):
  # max / min intensity plot
  # draw graph initial graph
  fig = plt.figure() #figsize=(width*10, height*10))
  for i in range(height * width):
    plt.subplot(height, width, i + 1)
    plt.tight_layout()
    if opt['im_dataset'] == 'MNIST':
      A = paths[i, :, :].cpu()
      plt.plot(torch.max(A,dim=1)[0], color='red')
      plt.plot(torch.min(A,dim=1)[0], color='green')
      plt.plot(torch.mean(A,dim=1), color='blue')
    elif opt['im_dataset'] == 'CIFAR':
      A = paths[i,:,:].view(paths.shape[1], opt['im_height'] * opt['im_width'], opt['im_chan']).cpu()
      plt.plot(torch.max(A, dim=1)[0][:,0],color='red')
      plt.plot(torch.max(A, dim=1)[0][:,1],color='green')
      plt.plot(torch.max(A, dim=1)[0][:,2],color='blue')
      plt.plot(torch.min(A, dim=1)[0][:,0],color='red')
      plt.plot(torch.min(A, dim=1)[0][:,1],color='green')
      plt.plot(torch.min(A, dim=1)[0][:,2],color='blue')
      plt.plot(torch.mean(A, dim=1)[:,0],color='red')
      plt.plot(torch.mean(A, dim=1)[:,1],color='green')
      plt.plot(torch.mean(A, dim=1)[:,2],color='blue')
    plt.title("Evolution of Pixel Intensity, Ground Truth: {}".format(labels[i].item()))
  return fig


@torch.no_grad()
def plot_att_heat(model, model_key, modelpath):
  pass
  # #visualisation of ATT for the 1st image in the batch
  # im_height = model.opt['im_height']
  # im_width = model.opt['im_width']
  # im_chan = model.opt['im_chan']
  # hwc = im_height * im_width
  # edge_index = model.odeblock.odefunc.edge_index
  # num_nodes = model.opt['num_nodes']
  # batch_size = model.opt['batch_size']
  # edge_weight = model.odeblock.odefunc.edge_weight
  # dense_att = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight,
  #                          max_num_nodes=num_nodes*batch_size)[0,:num_nodes,:num_nodes]
  # square_att = dense_att.view(num_nodes, num_nodes)
  # x_np = square_att.numpy()
  # x_df = pd.DataFrame(x_np)
  # x_df.to_csv(f"{modelpath}_att.csv")
  # fig = plt.figure()
  # plt.tight_layout()
  # plt.imshow(square_att, cmap='hot', interpolation='nearest')
  # plt.title("Attention Heat Map {}".format(model_key))
  # return fig
  # useful code to overcome normalisation colour bar
  # https: // matplotlib.org / 3.3.3 / gallery / images_contours_and_fields / multi_image.html  # sphx-glr-gallery-images-contours-and-fields-multi-image-py


@torch.no_grad()
def plot_image(paths, labels, time, opt, pic_folder, samples):
  savefolder = f"{pic_folder}/image_{time}"
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
  for i in range(samples):
    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    A = paths[i,time,:].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title(f"t={time} Ground Truth: {labels[i].item()}")
    plt.savefig(f"{savefolder}/image_{time}_{i}.png", format="png")
    plt.savefig(f"{savefolder}/image_{time}_{i}.pdf", format="pdf")
  return fig


@torch.no_grad()
def create_animation(paths, labels, frames, fps, opt, pic_folder, samples):
  savefolder = f"{pic_folder}/animations"
  try:
    os.mkdir(savefolder)
  except OSError:
    if os.path.exists(savefolder):
      shutil.rmtree(savefolder)
      os.mkdir(savefolder)
      print("%s exists, clearing existing images" % savefolder)
    else:
      print("Creation of the directory %s failed" % savefolder)
  # draw graph initial graph
  for i in range(samples):
    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    A = paths[i,0,:].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()

    if opt['im_dataset'] == 'MNIST':
      plt.imshow(A, cmap='gray', interpolation = 'none')
    elif opt['im_dataset'] == 'CIFAR':
      A = UnNormalizeCIFAR(A)
      plt.imshow(A, interpolation = 'none')
    plt.title("t=0 Ground Truth: {}".format(labels[i].item()))
    # loop through data and update plot
    def update(ii):
      plt.tight_layout()
      A = paths[i,ii,:].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
      if opt['im_dataset'] == 'MNIST':
        plt.imshow(A, cmap='gray', interpolation = 'none')
      elif opt['im_dataset'] == 'CIFAR':
        A = UnNormalizeCIFAR(A)
        plt.imshow(A, interpolation = 'none')
      plt.title(f"t={ii} Ground Truth: {labels[i].item()}")
    fig = plt.gcf()
    animation = FuncAnimation(fig, func=update, frames=frames)
    animation.save(f'{savefolder}/animation{i}.gif', fps=fps)#, writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=fps)
  # return animation


@torch.no_grad()
def create_pixel_intensity(paths, labels, opt, pic_folder, samples):
  savefolder = f"{pic_folder}/maxmin"
  try:
    os.mkdir(savefolder)
  except OSError:
    if os.path.exists(savefolder):
      shutil.rmtree(savefolder)
      os.mkdir(savefolder)
      print("%s exists, clearing existing images" % savefolder)
    else:
      print("Creation of the directory %s failed" % savefolder)
  for i in range(samples):
    fig = plt.figure()
    plt.tight_layout()
    plt.axis('off')
    if opt['im_dataset'] == 'MNIST':
      A = paths[i, :, :].cpu()
      plt.plot(torch.max(A, dim=1)[0], color='red')
      plt.plot(torch.min(A, dim=1)[0], color='green')
      plt.plot(torch.mean(A, dim=1), color='blue')
    elif opt['im_dataset'] == 'CIFAR':
      A = paths[i, :, :].view(paths.shape[1], opt['im_height'] * opt['im_width'], opt['im_chan']).cpu()
      plt.plot(torch.max(A, dim=1)[0][:, 0], color='red')
      plt.plot(torch.max(A, dim=1)[0][:, 1], color='green')
      plt.plot(torch.max(A, dim=1)[0][:, 2], color='blue')
      plt.plot(torch.min(A, dim=1)[0][:, 0], color='red')
      plt.plot(torch.min(A, dim=1)[0][:, 1], color='green')
      plt.plot(torch.min(A, dim=1)[0][:, 2], color='blue')
      plt.plot(torch.mean(A, dim=1)[:, 0], color='red')
      plt.plot(torch.mean(A, dim=1)[:, 1], color='green')
      plt.plot(torch.mean(A, dim=1)[:, 2], color='blue')
    plt.title("Max/Min, Ground Truth: {}".format(labels[i].item()))
    plt.savefig(f"{savefolder}/max_min_{i}.png", format="png")
    plt.savefig(f"{savefolder}/max_min_{i}.pdf", format="pdf")
  return fig


def get_paths(modelpath, model_key, opt, Tmultiple, partitions, batch_num=0):
  path_folder = f"../paths/{model_key}"
  if not os.path.exists(f"{path_folder}/{model_key}_Tx{Tmultiple}_{partitions}_paths.pt"):
    try:
      os.mkdir(path_folder)
    except OSError:
      print("Creation of the directory %s failed" % path_folder)

    #load data and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_train, data_test = load_data(opt)
    loader = DataLoader(data_train, batch_size=opt['batch_size'], shuffle=False)  # True)
    for batch_idx, batch in enumerate(loader):
      break
    batch.to(device)
    edge_index_gpu = batch.edge_index
    edge_attr_gpu = batch.edge_attr
    if edge_index_gpu is not None: edge_index_gpu.to(device)
    if edge_attr_gpu is not None: edge_index_gpu.to(device)
    opt['time'] = opt['time'] / partitions
    model = GNN_image_pixel(opt, batch.num_features, batch.num_nodes, opt['num_class'], edge_index_gpu,
                      batch.edge_attr, device).to(device)

    # # original saved file with DataParallel
    # state_dict = torch.load(modelpath, map_location=device)
    # # create new OrderedDict that does not contain module.
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #   name = k.replace(".module", "")  # removing ‘.moldule’ from key
    #   # name = k[7:]  # remove module.
    #   new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)


    # broken
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.to(device)
    model.eval()
    ###do forward pass
    for batch_idx, batch in enumerate(loader):
      if batch_idx == batch_num:
        paths = model.forward_plot_path(batch.x.to(model.device), Tmultiple * partitions)
        labels = batch.y
        break
    torch.save(paths,f"{path_folder}/{model_key}_Tx{Tmultiple}_{partitions}_paths.pt")
    torch.save(labels, f"{path_folder}/{model_key}_Tx{Tmultiple}_{partitions}_y.pt")
  else:
    paths = torch.load(f"{path_folder}/{model_key}_Tx{Tmultiple}_{partitions}_paths.pt")
    labels = torch.load(f"{path_folder}/{model_key}_Tx{Tmultiple}_{partitions}_y.pt")
  # paths.cpu().requires_grad = False
  # labels.cpu().requires_grad = False
  paths_nograd = paths.cpu().detach()
  labels_nograd = labels.cpu().detach()
  # paths.cpu().no_grad()
  # labels.cpu().no_grad()
  return paths_nograd, labels_nograd


def single_images(model_keys, samples, Tmultiple, partitions, batch_num):
  for model_key in model_keys:
    directory = f"../pixels/"
    for filename in os.listdir(directory):
      if filename.startswith(model_key):
        path = os.path.join(directory, filename)
        print(path)
        break
    [_, _, data_name, blck, fct] = path.split("_")

    modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
    modelpath = f"{modelfolder}/model_{model_key}"
    df = pd.read_csv(f'{directory}models.csv')
    optdf = df[df.model_key == model_key]
    intcols = ['num_class','im_chan','im_height','im_width','num_nodes']
    optdf[intcols].astype(int)
    opt = optdf.to_dict('records')[0]

    paths, labels = get_paths(modelpath, model_key, opt, Tmultiple, partitions, batch_num)

    # # 1)
    T_idx = Tmultiple * partitions // 2
    fig = plot_image_T(paths, labels, T_idx, opt, height=2, width=3)
    plt.savefig(f"{modelpath}_imageT.png", format="png")
    plt.savefig(f"{modelpath}_imageT.pdf", format="pdf")
    # 2)
    animation = create_animation_old(paths, labels, opt, height=2, width=3, frames=partitions)
    # animation.save(f'{modelpath}_animation.gif', writer='imagemagick', savefig_kwargs={'facecolor': 'white'}, fps=2)
    animation.save(f'{modelpath}_animation.gif', fps=2)

    # from IPython.display import HTML
    # HTML(animation.to_html5_video())
    # plt.rcParams['animation.ffmpeg_path'] = '/home/jr1419home/anaconda3/envs/GNN_WSL/bin/ffmpeg'
    # animation.save(f'{modelpath}_animation3.mp4', writer='ffmpeg', fps=2)
    # 3)
    # fig = plot_att_heat(model, model_key, modelpath)
    # plt.savefig(f"{modelpath}_AttHeat.pdf", format="pdf")
    # # 4)
    fig = create_pixel_intensity_old(paths, labels, opt, height=2, width=3)
    plt.savefig(f"{modelpath}_pixel_intensity.png", format="png")
    plt.savefig(f"{modelpath}_pixel_intensity.pdf", format="pdf")


def build_all(model_keys, samples, Tmultiple, partitions, batch_num):
  directory = f"../pixels/"
  df = pd.read_csv(f'{directory}models.csv')
  for model_key in model_keys:
    for filename in os.listdir(directory):
      if filename.startswith(model_key):
        path = os.path.join(directory, filename)
        print(path)
        break
    [_, _, data_name, blck, fct] = path.split("_")
    modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
    modelpath = f"{modelfolder}/model_{model_key}"
    optdf = df[df.model_key == model_key]
    intcols = ['num_class','im_chan','im_height','im_width','num_nodes']
    optdf[intcols].astype(int)
    opt = optdf.to_dict('records')[0]

    paths, labels = get_paths(modelpath, model_key, opt, Tmultiple, partitions, batch_num)

    plot_image(paths, labels, time=0, opt=opt, pic_folder=modelfolder, samples=samples)
    plot_image(paths, labels, time=5, opt=opt, pic_folder=modelfolder, samples=samples)
    plot_image(paths, labels, time=10, opt=opt, pic_folder=modelfolder, samples=samples)
    create_animation(paths, labels, Tmultiple*partitions, fps=2, opt=opt, pic_folder=modelfolder, samples=samples)
    create_pixel_intensity(paths, labels, opt, pic_folder=modelfolder, samples=samples)

@torch.no_grad()
def create_grid(grid_keys, times, sample_name, samples, Tmultiple, partitions, batch_num):
    directory = f"../pixels/"
    df = pd.read_csv(f'{directory}models.csv')
    savefolder = f"../pixels/images/{sample_name}"
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

    plot_times = [f"t={time/partitions}" for time in times]
    for sample in range(samples):
      images_A = []
      images_B = []
      images_C = []
      labels = []
      datasets = []
      for model_key in grid_keys:
        for filename in os.listdir(directory):
          if filename.startswith(model_key):
            path = os.path.join(directory, filename)
            print(path)
            break
        [_, _, data_name, blck, fct] = path.split("_")
        modelfolder = f"{directory}{model_key}_{data_name}_{blck}_{fct}"
        modelpath = f"{modelfolder}/model_{model_key}"
        optdf = df[df.model_key == model_key]
        intcols = ['num_class', 'im_chan', 'im_height', 'im_width', 'num_nodes']
        optdf[intcols].astype(int)
        opt = optdf.to_dict('records')[0]

        paths, _ = get_paths(modelpath, model_key, opt, Tmultiple, partitions, batch_num)

        A = paths[sample, times[0], :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
        B = paths[sample, times[1], :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
        C = paths[sample, times[2], :].view(opt['im_height'], opt['im_width'], opt['im_chan']).cpu()
        if opt['im_dataset'] == 'MNIST':
          images_A.append(A)
          images_B.append(B)
          images_C.append(C)
        elif opt['im_dataset'] == 'CIFAR':
          images_A.append(UnNormalizeCIFAR(A))
          images_B.append(UnNormalizeCIFAR(B))
          images_C.append(UnNormalizeCIFAR(C))
        labels.append(f"{opt['block']}\n{opt['function']}")
        datasets.append(opt['im_dataset'])
      images = [images_A, images_B, images_C]

      fig, axs = plt.subplots(3,3, figsize=(9, 6), sharex=True, sharey=True)
      fig.suptitle(f"{opt['im_dataset']} Pixel Diffusion")
      for i in range(3):
          for j in range(3):
              # axs[i,j].imshow(plt.imread(images[j][i]))
              if datasets[i] == 'MNIST':
                axs[i, j].imshow(images[j][i], cmap='gray', interpolation='none')
              elif datasets[i] == 'CIFAR':
                A = UnNormalizeCIFAR(A)
                axs[i, j].imshow(images[j][i], interpolation='none')
              axs[i,j].set_yticks([])
              axs[i,j].set_xticks([])
      plt.subplots_adjust(wspace=0, hspace=0)
      for ax, t in zip(axs[0], plot_times):
          ax.set_title(t, size=18)
      pad = 2
      for ax, row in zip(axs[:,0], labels):
          ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                      xycoords=ax.yaxis.label, textcoords='offset points',
                      size='large', ha='right', va='center')
      plt.savefig(f"{savefolder}/sample_{sample}.pdf",format="pdf")

def main(model_keys):
  pass

if __name__ == '__main__':
  Tmultiple = 2
  partitions = 10
  batch_num = 2
  samples = 6
  # model_keys = [
  # '20210127_015525',
  # '20210127_021404',
  # '20210127_043024',
  # '20210127_074633',
  # '20210127_044929',
  # '20210127_051136']
  model_keys = [
'20210129_013725',
'20210129_013907',
'20210129_014448',
'20210129_015003',
]
  #
  # model_keys = ['20210125_002603']
  # directory = f"../pixels/"
  # df = pd.read_csv(f'{directory}models.csv')
  # model_keys = df['model_key'].to_list()
  # model_keys = ['20210125_002603',
  #   '20210125_111920',
  #   '20210125_115601']
  #
  single_images(model_keys, samples, Tmultiple, partitions, batch_num)
  build_all(model_keys, samples, Tmultiple, partitions, batch_num)

  times = [0, 10, 20]

  grid_keys = [
  '20210129_013725',
  '20210129_013907',
  '20210129_013907']
  image_folder = 'TestPixel3_binary'
  create_grid(grid_keys, times, image_folder, samples, Tmultiple, partitions, batch_num)

  grid_keys = [
  '20210129_014448',
  '20210129_015003',
  '20210129_015003']
  image_folder = 'TestPixel4_binary'
  create_grid(grid_keys, times, image_folder, samples, Tmultiple, partitions, batch_num)
  # image_folder = 'MNIST1'
  # image_folder = 'CIFAR1'
  # grid_keys = ['20210127_074633','20210127_044929','20210127_051136']
  # create_grid(grid_keys, times, image_folder, samples, Tmultiple, partitions, batch_num)
