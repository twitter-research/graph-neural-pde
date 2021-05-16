import argparse
from ray.tune import Analysis
import json
import os
import pandas as pd
import numpy as np
from utils import get_sem, mean_confidence_interval
from ray_tune_rewiring import train_ray_int
from ray import tune
from functools import partial
import os, time
from ray.tune import CLIReporter


def appendDFToCSV_void(df, csvFilePath, sep=","):
  import os
  if not os.path.isfile(csvFilePath):
    df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
  elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
    raise Exception("Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
      len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
  elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
    raise Exception("Columns and column order of dataframe and csv file do not match!!")
  else:
    df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)

def get_best_params_dir(opt):
  analysis = Analysis("../ray_tune/{}".format(opt['folder']))
  df = analysis.dataframe(metric=opt['metric'], mode='max')
  best_params_dir = df.sort_values(opt['metric'], ascending=False)['logdir'].iloc[opt['index']]
  return best_params_dir

def with_KNN(opt):
  opt['rewire_KNN'] = True
  opt['rewire_KNN_T'] = "T0"
  opt['rewire_KNN_epoch'] = 20
  opt['rewire_KNN_k'] = 64
  opt['rewire_KNN_sym'] = False
  return opt


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
  # handle adjoint
  if best_params['adjoint'] or opt['adjoint']:
    best_params_ret['adjoint'] = True

  if opt["run_with_KNN"]:
    best_params_ret = with_KNN(best_params_ret)

  if opt['change_att_sim_type']:
    best_params_ret['attention_type'] = opt['att_sim_type']
    best_params_ret['square_plus'] = False

  best_params_ret['pos_enc_orientation'] = best_params_ret['pos_enc_dim']

  print("Running with parameters {}".format(best_params_ret))

  data_dir = os.path.abspath("../data")
  reporter = CLIReporter(
    metric_columns=["accuracy", "loss", "test_acc", "train_acc", "best_time", "best_epoch", "training_iteration", "forward_nfe", "backward_nfe"])

  if opt['name'] is None:
    name = opt['folder'] + '_test'
  else:
    name = opt['name']

  result = tune.run(
    partial(train_ray_int, data_dir=data_dir),
    name=name,
    resources_per_trial={"cpu": opt['cpus'], "gpu": opt['gpus']},
    search_alg=None,
    keep_checkpoints_num=3,
    checkpoint_score_attr='accuracy',
    config=best_params_ret,
    num_samples=opt['reps'] if opt["num_splits"] == 0 else opt["num_splits"] * opt["reps"],
    scheduler=None,
    max_failures=1,  # early stop solver can't recover from failure as it doesn't own m2.
    local_dir='../ray_tune',
    progress_reporter=reporter,
    raise_on_failed_trial=False)

  df = result.dataframe(metric=opt['metric'], mode="max").sort_values(opt['metric'], ascending=False)

  try:
    csvFilePath = '../ray_results/{}.csv'.format(name)  # , time.strftime("%Y%m%d-%H%M%S"))
    appendDFToCSV_void(df, csvFilePath, sep=",")
    # df.to_csv('../ray_results/{}_{}.csv'.format(name, time.strftime("%Y%m%d-%H%M%S")))
  except:
    pass

  print(df[['accuracy', 'test_acc', 'train_acc', 'best_time', 'best_epoch']])

  test_accs = df['test_acc'].values
  print("test accuracy {}".format(test_accs))
  log = "mean test {:04f}, test std {:04f}, test sem {:04f}, test 95% conf {:04f}"
  print(log.format(test_accs.mean(), np.std(test_accs), get_sem(test_accs), mean_confidence_interval(test_accs)))


def mainLoop(opt):
  datas = ['Cora','Citeseer'] #, 'Pubmed'] #['Cora', 'Citeseer', 'Photo']
  folders = ['beltrami_2','Citeseer_beltrami_1']#, 'Pubmed_beltrami_2_KNN'] #['Cora_beltrami_1_KNN', 'Citeseer_beltrami_1_KNN', 'Photo_beltrami_1_KNN']
  names = ['Cora_beltrami_attdefaults_test','Citeseer_beltrami_attdefaults_test']

  indexes = [[3,4],[3,4]] #[[0,1,2], [0,1,2]] #,3,4]]#, [0,1,2,3,4]] #,0,0]

  opt['run_with_KNN'] = False

  opt['change_att_sim_type'] = False
  opt['bestwithAttTypes'] = ['cosine_sim', 'scaled_dot'] #[False]

  for i, ds in enumerate(datas):
    for idx_i, idx in enumerate(indexes[i]):
      if opt['change_att_sim_type']:
        for att_type in opt['bestwithAttTypes']:
          print(f"Running Best Params for {ds}")
          opt["dataset"] = ds
          opt["folder"] = folders[i]
          opt["name"] = f"{names[i]}{'_KNN' if opt['bestwithKNN'] else ''}"
          opt["index"] = indexes[i][idx_i]
          opt['att_sim_type'] = att_type
          run_best_params(opt)
      else:
        print(f"Running Best Params for {ds}")
        opt["dataset"] = ds
        opt["folder"] = folders[i]
        opt["name"] = f"{names[i]}{'_KNN' if opt['bestwithKNN'] else ''}"
        opt["index"] = indexes[i][idx_i]
        run_best_params(opt)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--epoch', type=int, default=10, help='Number of training epochs per iteration.')
  parser.add_argument('--folder', type=str, default=None, help='experiment folder to read')
  parser.add_argument('--index', type=int, default=0, help='index to take from experiment folder')
  parser.add_argument('--metric', type=str, default='accuracy', help='metric to sort the hyperparameter tuning runs on')
  parser.add_argument('--augment', action='store_true',
                      help='double the length of the feature vector by appending zeros to stabilise ODE learning')
  parser.add_argument('--run_with_KNN', action='store_true',
                      help='run parameters discovered without KNN, but now using KNN')
  parser.add_argument('--change_att_sim_type', action='store_true',
                      help='run parameters discovered with attention different attention similarity ')
  parser.add_argument('--reps', type=int, default=1, help='the number of random weight initialisations to use')
  parser.add_argument('--name', type=str, default=None)
  #todo if we add new pos_enc_types this will break
  parser.add_argument('--pos_enc_type', type=str, default="GDC", help='positional encoder (default: GDC)')
  parser.add_argument('--gpus', type=float, default=0, help='number of gpus per trial. Can be fractional')
  parser.add_argument('--cpus', type=float, default=1, help='number of cpus per trial. Can be fractional')
  parser.add_argument("--num_splits", type=int, default=0, help="Number of random slpits >= 0. 0 for planetoid split")
  parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                      help="use the adjoint ODE method to reduce memory footprint")
  parser.add_argument("--max_nfe", type=int, default=5000, help="Maximum number of function evaluations allowed.")
  parser.add_argument("--no_early", action="store_true",
                      help="Whether or not to use early stopping of the ODE integrator when testing.")

  parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')

  args = parser.parse_args()

  opt = vars(args)
  run_best_params(opt)
  # mainLoop(opt)