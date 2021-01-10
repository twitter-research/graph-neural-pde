"""
As population based training generates a path through the hyperparameter space it is necessary to have a custom
module to read the hyperparameter path (the policy) and replay it
"""
import os
from ray import tune
from ray.tune.schedulers import PopulationBasedTrainingReplay
from run_best_ray import get_best_params_dir
from functools import partial
from ray_tune import train_ray
import argparse

# replay = PopulationBasedTrainingReplay(
#     "~/ray_results/pbt_test/pbt_policy_ba982_00003.txt")

def get_replay_str(folder, idx):
  dir_name = get_best_params_dir(folder, idx)
  dir_name_split = dir_name.split('_')
  replay_str = "pbt_policy_{}_{}.txt".format(dir_name_split[2], dir_name_split[3])
  return replay_str

def run_best_policy(opt):
  data_dir = os.path.abspath("../data")
  replay_str = get_replay_str(opt['folder'], opt['index'])
  local_dir = '../ray_tune'
  replay_path = "{}/{}/{}".format(local_dir, opt['folder'], replay_str)
  replay = PopulationBasedTrainingReplay(replay_path)
  tune.run(
      partial(train_ray, data_dir=data_dir),
      scheduler=replay,
      stop={"training_iteration": 100})

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--folder', type=str, default=None, help='experiment folder to read')
  parser.add_argument('--index', type=int, default=0, help='index to take from experiment folder')
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                      help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilise ODE learning')
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default='ray_test')
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  args = parser.parse_args()

  opt = vars(args)

  run_best_policy(opt)
