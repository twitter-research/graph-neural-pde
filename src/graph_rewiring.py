"""
functions to generate a graph from the input graph and features
"""

from torch_geometric.transforms.two_hop import TwoHop
from torch_geometric.transforms import GDC


def get_two_hop(data):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  th = TwoHop()
  data = th(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data


def apply_gdc(data, opt):
  print('raw data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  print('performing gdc transformation with method {}, sparsification {}'.format(opt['gdc_method'], opt['gdc_sparsification']))
  if opt['gdc_method'] == 'ppr':
    diff_args = dict(method='ppr', alpha=opt['ppr_alpha'])
  else:
    diff_args = dict(method='heat', t=opt['heat_time'])
  if opt['gdc_sparsification'] == 'topk':
    sparse_args = dict(method='topk', k=opt['gdc_k'], dim=0)
  else:
    sparse_args = dict(method='threshold', eps=opt['gdc_threshold'])
    diff_args['eps'] = opt['gdc_threshold']
  print('gdc sparse args: {}'.format(sparse_args))
  gdc = GDC(float(opt['self_loop_weight']), normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=diff_args,
            sparsification_kwargs=sparse_args, exact=opt['exact'])
  if isinstance(data.num_nodes, list):
    data.num_nodes = data.num_nodes[0]
  data = gdc(data)
  print('following rewiring data contains {} edges and {} nodes'.format(data.num_edges, data.num_nodes))
  return data
