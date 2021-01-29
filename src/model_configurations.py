from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_dorsey_attention import ODEFuncDorseyAtt
from function_laplacian_diffusion import LaplacianODEFunc
from sde import SDEFunc, SDEblock
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from block_mixed import MixedODEblock
from block_transformer_hard_attention import HardAttODEblock

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
  elif ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'SDE':
    block = SDEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'SDE':
    f = SDEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'dorsey':
    f = ODEFuncDorseyAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  else:
    raise FunctionNotDefined
  return f
