# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from lib.utils import get_logger, count_parameters
from lib.data_utils import load_data
from lib.models import load_model
import torch
import datetime
import numpy as np
import sys
import os

parser = argparse.ArgumentParser('CRU')
# train configs
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs.")
parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
parser.add_argument('--lr-decay', type=float, default=1, help="Learning rate decay.")
parser.add_argument('--weight-decay', type=float, default=0, help="Weight decay.")
parser.add_argument('-b', '--batch-size', type=int, default=50, help="Batch size for training and test set.")
parser.add_argument('--grad-clip', action='store_true', help="If to use gradient clipping.")
parser.add_argument('--ts', type=float, default=1, help="Scaling factor of timestamps for numerical stability.")
parser.add_argument('--save-intermediates', type=str, default=None,
                    help="Directory path for saving model intermediates (post_mean, post_cov, prior_mean, prior_cov, kalman_gain, y, y_var). If None, no intermediates are saved.")
parser.add_argument('--log-rythm', type=int, default=20,
                    help="Save heatmaps of model intermediates to tensorboard every log-rythm epoch. Ignored if tensorboard not used.")
parser.add_argument('--task', type=str,
                    help="Possible tasks are interpolation, extrapolation, regression, one_step_ahead_prediction.")
parser.add_argument('--anomaly-detection', action='store_true',
                    help="If to trace NaN values in backpropagation for debugging.")
parser.add_argument('--tensorboard', action='store_true',
                    help="If to use tensorboard for logging additional to standard logger.")
# CRU transition model 
parser.add_argument('-lsd', '--latent-state-dim', type=int, default=None,
                    help="Latent state dimension. Accepts only even values because latent observation dimenions = latent state dimension / 2")
parser.add_argument('--hidden-units', type=int, default=50, help="Hidden units of encoder and decoder.")
parser.add_argument('--num-basis', type=int, default=15,
                    help="Number of basis matrices to use in transition model for locally-linear transitions. K in paper")
parser.add_argument('--bandwidth', type=int, default=3, help="Bandwidth for basis matrices A_k. b in paper")
parser.add_argument('--enc-var-activation', type=str, default='elup1',
                    help="Variance activation function in encoder. Possible values elup1, exp, relu, square")
parser.add_argument('--dec-var-activation', type=str, default='elup1',
                    help="Variance activation function in decoder. Possible values elup1, exp, relu, square")
parser.add_argument('--trans-net-hidden-activation', type=str, default='tanh',
                    help="Activation function for transition net.")
parser.add_argument('--trans-net-hidden-units', type=list, default=[], help="Hidden units of transition net.")
parser.add_argument('--trans-var-activation', type=str, default='elup1', help="Activation function for transition net.")
parser.add_argument('--learn-trans-covar', type=bool, default=True, help="If to learn transition covariance.")
parser.add_argument('--learn-initial-state-covar', action='store_true',
                    help="If to learn the initial state covariance.")
parser.add_argument('--initial-state-covar', type=int, default=1, help="Value of initial state covariance.")
parser.add_argument('--trans-covar', type=float, default=0.1, help="Value of initial transition covariance.")
parser.add_argument('--t-sensitive-trans-net', action='store_true',
                    help="If to provide the time gap as additional input to the transition net. Used for RKN-Delta_t model in paper")
parser.add_argument('--f-cru', action='store_true',
                    help="If to use fast transitions based on eigendecomposition of the state transitions (f-CRU variant).")
parser.add_argument('--rkn', action='store_true', help="If to use discrete state transitions (RKN baseline).")
parser.add_argument('--orthogonal', type=bool, default=True,
                    help="If to use orthogonal basis matrices in the f-CRU variant.")
# data configs
parser.add_argument('--dataset', type=str, default=None,
                    help="Dataset to use. Available datasets are physionet, ushcn and pendulum.")
parser.add_argument('--sample-rate', type=float, default=1,
                    help='Sample time points to increase irregularity of timestamps. For example, if sample_rate=0.5 half of the time points are discarded at random in the data preprocessing.')
parser.add_argument('--impute-rate', type=float, default=None,
                    help='Remove time points for interpolation. For example, if impute_rate=0.3 the model is given 70% of the time points and tasked to reconstruct the entire series.')
parser.add_argument('--unobserved-rate', type=float, default=0.2,
                    help='Percentage of features to remove per timestamp in order to increase sparseness across dimensions (applied only for USHCN)')
parser.add_argument('--cut-time', type=int, default=None, help='Timepoint at which extrapolation starts.')
parser.add_argument('--num-workers', type=int, default=8, help="Number of workers to use in dataloader.")
parser.add_argument('--pin-memory', type=bool, default=True, help="If to pin memory in dataloader.")
parser.add_argument('--data-random-seed', type=int, default=0,
                    help="Random seed for subsampling timepoints and features.")
parser.add_argument('-rs', '--random-seed', type=int, default=0, help="Random seed for initializing model parameters.")
parser.add_argument('-np', '--num_past', type=int, default=None, help="Number of observations to use for training.")
parser.add_argument('-nf', '--num_future', type=int, default=None,
                    help="Number of observations to use as ground truth.")

parser.add_argument('--timestamp_freq', type=str, default='M', help="Frequency of timestamps in expanding data.")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
identifier = datetime.datetime.now().strftime("%m-%d--%H-%M-%S")

if __name__ == '__main__':

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)

    log_path = os.path.join("logs", args.dataset, args.task + '_' + identifier + ".log")
    if not os.path.exists(f"logs/{args.dataset}"):
        os.makedirs(f"logs/{args.dataset}")

    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    train_dl, valid_dl = load_data(args)
    model = load_model(args)
    logger.info(f'parameters: {count_parameters(model)}')

    model.train(train_dl=train_dl, valid_dl=valid_dl,
                identifier=identifier, logger=logger)
