from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from GNN import GNN
from run_GNN import train, test, get_optimizer, get_cora_opt
from data import get_dataset
import torch
from torch import nn
import time
import os
import argparse
import numpy as np
from ray_tune import train_ray


def set_pop_search_space(opt):
  opt['decay'] = tune.loguniform(2e-3, 5e-2)
  opt['hidden_dim'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 7))
  opt['lr'] = tune.loguniform(0.001, 0.03)
  opt['input_dropout'] = tune.uniform(0, 0.8)
  opt['dropout'] = tune.uniform(0, 0.8)
  if opt['ode'] == 'att':
    opt['self_loop_weight'] = tune.choice([0,1])
  else:
    opt['self_loop_weight'] = tune.uniform(0, 5)
  opt['time'] = tune.uniform(1., 10.)
  opt['tol_scale'] = tune.loguniform(1e1, 1e5)
  return opt


class CustomStopper(tune.Stopper):
  def __init__(self, max_iter):
    self.should_stop = False
    self.max_iter = max_iter

  def __call__(self, trial_id, result):
    if not self.should_stop and result["accuracy"] > 0.96:
      self.should_stop = True
    return self.should_stop or result["training_iteration"] >= self.max_iter

  def stop_all(self):
    return self.should_stop


def main(opt):
  data_dir = os.path.abspath("../data")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  opt = set_pop_search_space(opt)
  scheduler = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='accuracy',
    mode='max',
    resample_probability=0.75,
    perturbation_interval=opt['pi'],
    hyperparam_mutations={
      'lr': tune.loguniform(0.001, 0.03),
      'self_loop_weight': tune.uniform(0, 5),
      'decay': tune.loguniform(2e-3, 5e-2),
      'input_dropout': tune.uniform(0, 0.8),
      'dropout': tune.uniform(0, 0.8),
      # 'hidden_dim' : tune.uniform(16, 64),
      'tol_scale': tune.loguniform(1e2, 1e4),
      'time': tune.uniform(1., 15.),
      # 'alpha_dim': ['sc', 'vc'],
      'no_alpha_sigmoid': [True, False]
    })
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "training_iteration"])
  result = tune.run(
    partial(train_ray, data_dir=data_dir),
    name=opt['name'],
    stop=CustomStopper(opt['max_iter']),
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    config=opt,
    num_samples=opt['num_samples'],
    scheduler=scheduler,
    max_failures=3,
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  best_trial = result.get_best_trial("accuracy", "max", "all")
  print("Best trial config: {}".format(best_trial.config))
  print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
  print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

  dataset = get_dataset(opt, data_dir, False)
  best_trained_model = GNN(best_trial.config, dataset, device)
  if opt['gpus'] > 1:
    best_trained_model = nn.DataParallel(best_trained_model)
  best_trained_model.to(device)

  checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

  model_state, optimizer_state = torch.load(checkpoint_path)
  best_trained_model.load_state_dict(model_state)

  test_acc = test(best_trained_model, best_trained_model.data.to(device))
  print("Best trial test set accuracy: {}".format(test_acc))
  df = result.dataframe(metric="accuracy", mode="max").sort_values('accuracy',
                                                                   ascending=False)  # get max accuracy for each trial
  timestr = time.strftime("%Y%m%d-%H%M%S")
  df.to_csv('../hyperopt_results/result_{}.csv'.format(timestr))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='Cora',
                      help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS')
  parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension.')
  parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
  parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
  parser.add_argument('--decay', type=float, default=0.005, help='Weight decay for optimization')
  parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
  parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs per iteration.')
  parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
  parser.add_argument('--time', type=float, default=7.0, help='End time of ODE function.')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilist ODE learning')
  parser.add_argument('--alpha_dim', type=str, default='vc', help='choose either scalar (sc) or vector (vc) alpha')
  parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true', help='apply sigmoid before multiplying by alpha')
  parser.add_argument('--beta_dim', type=str, default='vc', help='choose either scalar (sc) or vector (vc) beta')
  # ODE args
  parser.add_argument('--method', type=str, default='dopri5',
                      help="set the numerical solver: dopri5, euler, rk4, midpoint")
  parser.add_argument('--ode', type=str, default='att', help="set ode block. Either 'ode', 'att', 'sde'")
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument('--tol_scale', type=float, default=30., help='multiplier for atol and rtol')
  # SDE args
  parser.add_argument('--dt_min', type=float, default=1e-5, help='minimum timestep for the SDE solver')
  parser.add_argument('--dt', type=float, default=1e-3, help='fixed step size')
  parser.add_argument('--adaptive', type=bool, default=False, help='use adaptive step sizes')
  # Attention args
  parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
  parser.add_argument('--leaky_relu_slope', type=float, default=0.2, help='slope of the negative part of the leaky relu used in attention')
  parser.add_argument('--heads', type=int, default=5, help='number of attention heads')
  # ray args
  parser.add_argument('--num_samples', type=int, default=20, help='number of ray trials')
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument('--name', type=str, default='ray_exp')
  parser.add_argument('--max_iter', type=int, default=100,
                      help='maximum number of iterations of a population search trial')
  parser.add_argument('--pi', type=int, default=5, help='perturbation interval: the mutation frequency')

  args = parser.parse_args()

  opt = vars(args)

  main(opt)
