from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc
from function_greed import ODEFuncGreed
from function_greed_scaledDP import ODEFuncGreed_SDB
from function_greed_linear import ODEFuncGreedLinear
from function_greed_linear_homo import ODEFuncGreedLinH
from function_greed_linear_hetero import ODEFuncGreedLinHet
from function_transformer_attention_greed import ODEFuncTransformerAttGreed
from function_laplacian_diffusion_greed import LaplacianODEFunc_greed
from function_greed_non_linear import ODEFuncGreedNonLin
from block_transformer_attention import AttODEblock
from block_transformer_attention_greed import AttODEblock_greed
from block_constant import ConstantODEblock
from block_mixed import MixedODEblock
from block_transformer_hard_attention import HardAttODEblock
from block_transformer_rewiring import RewireAttODEblock
from block_greed_lie_trot import GREEDLTODEblock
class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'mixed':
    block = MixedODEblock
  elif ode_str == 'attention':
    block = AttODEblock
  elif ode_str == 'hard_attention':
    block = HardAttODEblock
  elif ode_str == 'rewire_attention':
    block = RewireAttODEblock
  elif ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'attention_greed':
    block = AttODEblock_greed
  elif ode_str == 'greed_lie_trotter':
    block = GREEDLTODEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'greed':
    f = ODEFuncGreed
  elif ode_str == 'greed_scaledDP':
    f = ODEFuncGreed_SDB
  elif ode_str == 'greed_linear':
    f = ODEFuncGreedLinear
  elif ode_str == 'greed_linear_homo':
    f = ODEFuncGreedLinH
  elif ode_str == 'greed_linear_hetero':
    f = ODEFuncGreedLinHet
  elif ode_str == 'transformer_greed':
    f = ODEFuncTransformerAttGreed
  elif ode_str == 'laplacian_greed':
    f = LaplacianODEFunc_greed
  elif ode_str == 'greed_non_linear':
    f = ODEFuncGreedNonLin
  elif ode_str == 'greed_lie_trotter':
    f = ODEFuncGreedLieTrot
  elif ode_str in ['gcn', 'gcn2', 'mlp', 'gcn_dgl' , 'gcn_res_dgl', 'gat']:
    f = ODEFuncGreedNonLin #hack required for code homogeniety
  else:
    raise FunctionNotDefined
  return f
