import argparse
from ray.tune import Analysis
import json
import numpy as np
from utils import get_sem, mean_confidence_interval
from ray_tune import train_ray_int
from ray import tune
from functools import partial
import os, time
from ray.tune import CLIReporter


def get_best_params_dir(opt):
  analysis = Analysis("../ray_tune/{}".format(opt['folder']))
  df = analysis.dataframe(metric='accuracy', mode='max')
  best_params_dir = df.sort_values('accuracy', ascending=False)['logdir'].iloc[opt['index']]
  return best_params_dir


def run_best_params(opt):
  best_params_dir = get_best_params_dir(opt)
  with open(best_params_dir + '/params.json') as f:
    best_params = json.loads(f.read())
  # allow params specified at the cmd line to override
  best_params_ret = {**best_params, **opt}
  try:
    best_params_ret['mix_features']
  except KeyError:
    best_params_ret['mix_features'] = False
  # the exception is number of epochs as we want to use more here than we would for hyperparameter tuning.
  best_params_ret['epoch'] = opt['epoch']
  best_params_ret['max_nfe'] = opt['max_nfe']

  print("Running with parameters {}".format(best_params_ret))

  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "training_iteration"])

  if opt['name'] is None:
    name = opt['folder'] + '_test'
  else:
    name = opt['name']

  result = tune.run(
    partial(train_ray_int, data_dir=data_dir),
    name=name,
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    search_alg=None,
    config=best_params_ret,
    num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
    scheduler=None,
    max_failures=0,
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric="accuracy", mode="max").sort_values('accuracy', ascending=False)
  try:
    df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
  except:
    pass

  print(df['accuracy'])

  test_accs = df['accuracy'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--folder', type=str, default=None, help='experiment folder to read')
  parser.add_argument('--index', type=int, default=0, help='index to take from experiment folder')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilise ODE learning')
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")
  parser.add_argument('--adjoint', default=False, help='use the adjoint ODE method to reduce memory footprint')
  parser.add_argument("--max_nfe", type=int, default=5000, help="Maximum number of function evaluations allowed.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")

  args = parser.parse_args()

  opt = vars(args)
  run_best_params(opt)
