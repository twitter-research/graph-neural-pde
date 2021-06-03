from function_transformer_attention import ODEFuncTransformerAtt
from function_laplacian_diffusion import LaplacianODEFunc
from block_transformer_attention import AttODEblock
from block_constant import ConstantODEblock
from block_transformer_hard_attention import HardAttODEblock

class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'attention':
    block = AttODEblock
  elif ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'hard_attention':
    block = HardAttODEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  else:
    raise FunctionNotDefined
  return f
