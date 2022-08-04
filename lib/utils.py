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
#
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

# This source code is derived from Latent ODEs for Irregularly-Sampled Time Series (https://github.com/YuliaRubanova/latent_ode)
# Copyright (c) 2019 Yulia Rubanova
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os 


# taken from https://github.com/ALRhub/rkn_share/ and not modified
class TimeDistributed(nn.Module):

    # taken from https://github.com/ALRhub/rkn_share/ and not modified
    def __init__(self, module, low_mem=False, num_outputs=1):
        """
        Makes a torch model time distributed. If the original model works with Tensors of size [batch_size] + data_shape
        this wrapper makes it work with Tensors of size [batch_size, sequence_length] + data_shape
        :param module: The module to wrap
        :param low_mem: Default is to the fast but high memory version. If you run out of memory set this to True
                        (it will be slower than)
            - low memory version: simple forloop over the time axis -> slower but consumes less memory
            - not low memory version: "reshape" and then process all at once -> faster but consumes more memory
        :param num_outputs: Number of outputs of the original module (really the number of outputs,
               not the dimensionality, e.g., for the normal RKN encoder that should be 2 (mean and variance))
        """

        super(TimeDistributed, self).__init__()
        self._module = module
        if num_outputs > 1:
            self.forward = self._forward_low_mem_multiple_outputs if low_mem else self._forward_multiple_outputs
        else:
            self.forward = self._forward_low_mem if low_mem else self._forward
        self._num_outputs = num_outputs

    def _forward(self, *args):
        input_shapes = [args[i].shape for i in range(len(args))]
        batch_size, seq_length = input_shapes[0][0], input_shapes[0][1]
        out = self._module(*[x.view(batch_size * seq_length,
                           *input_shapes[i][2:]) for i, x in enumerate(args)])
        return out.view(batch_size, seq_length, *out.shape[1:])

    def _forward_multiple_outputs(self, *args):
        input_shapes = [args[i].shape for i in range(len(args))]
        batch_size, seq_length = input_shapes[0][0], input_shapes[0][1]
        outs = self._module(
            *[x.view(batch_size * seq_length, *input_shapes[i][2:]) for i, x in enumerate(args)])
        out_shapes = [outs[i].shape for i in range(self._num_outputs)]
        return [outs[i].view(batch_size, seq_length, *out_shapes[i][1:]) for i in range(self._num_outputs)]

    def _forward_low_mem(self, x):
        out = []
        unbound_x = x.unbind(1)
        for x in unbound_x:
            out.append(self._module(x))
        return torch.stack(out, dim=1)

    def _forward_low_mem_multiple_outputs(self, x):
        out = [[] for _ in range(self._num_outputs)]
        unbound_x = x.unbind(1)
        for x in unbound_x:
            outs = self._module(x)
            [out[i].append(outs[i]) for i in range(self._num_outputs)]
        return [torch.stack(out[i], dim=1) for i in range(self._num_outputs)]


# taken from https://github.com/ALRhub/rkn_share/ and not modified
class MyLayerNorm2d(nn.Module):

    def __init__(self, channels):
        super(MyLayerNorm2d, self).__init__()
        self._scale = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self._offset = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        normalized = (x - x.mean(dim=[-3, -2, -1], keepdim=True)
                      ) / x.std(dim=[-3, -2, -1], keepdim=True)
        return self._scale * normalized + self._offset


# new code component
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# new code component
def update_learning_rate(optimizer, decay_rate=0.999, lowest=1e-3):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr


# new code component
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# taken from https://github.com/YuliaRubanova/latent_ode and not modified
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger


# new code component
def extract_intermediates(intermediates):
    post_mean = intermediates[0]
    post_u, post_l, post_s = intermediates[1]
    prior_mean = intermediates[2]
    prior_u, prior_l, prior_s = intermediates[3]
    q_u, q_l = intermediates[4]
    y = intermediates[5]
    y_var = intermediates[6]

    intermediates = [post_u, post_l, post_s, prior_u, prior_l,
                     prior_s, post_mean, prior_mean, q_u, q_l, y, y_var]
    intermediates_names = ['post_u', 'post_l', 'post_s', 'prior_u', 'prior_l',
                           'prior_s', 'post_mean', 'prior_mean', 'q_u', 'q_l', 'y', 'y_var']

    return intermediates, intermediates_names


# new code component
def log_to_tensorboard(model, writer, mode, metrics, output, input, intermediates, epoch, imput_metrics=None, log_rythm=20):
    ll, rmse, mse = metrics
    output_mean, output_var = output
    obs, truth, mask_obs = input
    mode_name = 'training' if mode == 'train' else 'validation'
    args = model.args
    matrix_list = 'exp_A' if args.f_cru else ['exp_A', 'exp_B',
                   'M2'] 

    writer.add_scalar(f'{mode_name} ll', ll, epoch)
    writer.add_scalar(f'{mode_name} rmse', rmse, epoch)
    writer.add_scalar(f'{mode_name} mse', mse, epoch)

    if imput_metrics is not None:
        writer.add_scalar(f'{mode_name} ll imput', imput_metrics[0], epoch)
        writer.add_scalar(f'{mode_name} mse imput', imput_metrics[1], epoch)

    #writer.add_scalar(f'{mode} variance', torch.sum(mask_truth * output_var), epoch)
    if epoch % log_rythm == 0:
        if mode == 'train':
            for matrix in matrix_list:
                f, ax = plt.subplots()
                sns.heatmap(model._cru_layer._cell.__dict__[
                            matrix].detach().cpu()[0, ...], ax=ax)
                writer.add_figure(matrix, f, epoch)

            for name, comp in intermediates.items():
                f, ax = plt.subplots()
                sns.heatmap(comp.detach().cpu()[0, ...], ax=ax)
                writer.add_figure(name, f, epoch)

            for name, par in model.named_parameters():
                if par.requires_grad:
                    writer.add_histogram(name, par, epoch)
                    writer.add_histogram('gradient:'+name, par.grad, epoch)

        writer.add_histogram(f'{mode} out_mean', output_mean, epoch)

        if model.bernoulli_output:
            for comp, name in zip([obs, truth, output_mean], ['obs', 'targets', 'out']):
                writer.add_images(f'{mode} {name}',
                                  comp[0, ...], epoch, dataformats='NCHW')

        elif args.task == 'regression':
            writer.add_images(
                f'{mode} obs', obs[0, ...], epoch, dataformats='NCHW')

        else:
            writer.add_histogram(f'{mode} out_var', output_var, epoch)
            for comp, name in zip([obs, mask_obs, truth, output_mean, output_var], ['obs', 'mask_obs', 'targets', 'out', 'out var']):
                f, ax = plt.subplots()
                sns.heatmap(comp.detach().cpu()[0, ...], ax=ax)
                writer.add_figure(f'{mode} {name}', f, epoch)
