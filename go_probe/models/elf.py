# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

class Model(nn.Module):
    ''' Base class for an RL model, it is a wrapper for ``nn.Module``'''

    def __init__(self, option_map, params):
        """Initialize model with ``args``.

        Set ``step`` to ``0`` and ``volatile`` to ```false``.

        ``step`` records the number of times the weight has been updated.
        ``volatile`` indicates that the Variable should be used in
        inference mode, i.e. don't save the history.
        """
        super(Model, self).__init__()
        self.option_map = option_map
        self.options = option_map
        self.params = params
        self.step = 0
        self.volatile = False

    def clone(self, gpu=None):
        """Deep copy an existing model.

        ``options``, ``step`` and ``state_dict`` are copied.

        Args:
            gpu(int): gpu id to be put the model on

        Returns:
            Cloned model
        """
        model = type(self)(self.option_map, self.params)
        model.load_state_dict(deepcopy(self.state_dict()))
        model.step = self.step
        if gpu is not None:
            model.cuda(gpu)
        return model

    def set_volatile(self, volatile):
        """Set model to ``volatile``.

        Args:
            volatile(bool): indicating that the Variable should be used in
                            inference mode, i.e. don't save the history.
        """
        self.volatile = volatile

    def _var(self, x):
        ''' Convert tensor x to a pytorch Variable.

        Returns:
            Variable for x
        '''
        if not isinstance(x, Variable):
            return Variable(x, volatile=self.volatile)
        else:
            return x

    def before_update(self):
        """Customized operations for each model before update.

        To be extended.

        """
        pass

    def save(self, filename, num_trial=10):
        """Save current model, step and args to ``filename``

        Args:
            filename(str): filename to be saved.
            num_trial(int): maximum number of retries to save a model.
        """
        # Avoid calling the constructor by doing self.clone()
        # deepcopy should do it
        state_dict = deepcopy(self).cpu().state_dict()

        # Note that the save might experience issues, so if we encounter
        # errors, try a few times and then give up.
        content = {
            'state_dict': state_dict,
            'step': self.step,
            'options': vars(self.options),
        }
        for i in range(num_trial):
            try:
                torch.save(content, filename)
                return
            except BaseException:
                sleep(1)
        print(
            "Failed to save %s after %d trials, giving up ..." %
            (filename, num_trial))

    def load(
            self, filename,
            omit_keys=[], replace_prefix=[], check_loaded_options=True):
        ''' Load current model, step and args from ``filename``

        Args:
            filename(str): model filename to load from
            omit_keys(list): list of omitted keys.
                             Sometimes model will have extra keys and weights
                             (e.g. due to extra tasks during training).
                             We should omit them;
                             otherwise loading will not work.
        '''
        data = torch.load(filename)

        if isinstance(data, OrderedDict):
            self.load_state_dict(data)
        else:
            for k in omit_keys:
                del data["state_dict"][k + ".weight"]
                del data["state_dict"][k + ".bias"]

            sd = data["state_dict"]

            keys = list(sd.keys())
            for key in keys:
                # Should be commented out for PyTorch > 0.40
                # if key.endswith("num_batches_tracked"):
                #    del sd[key]
                #     continue
                for src, dst in replace_prefix:
                    if key.startswith(src):
                        # print(f"Src=\"{src}\", Dst=\"{dst}\"")
                        sd[dst + key[len(src):]] = sd[key]
                        del sd[key]

            self.load_state_dict(sd)
        self.step = data.get("step", 0)
        self.filename = os.path.realpath(data.get("filename", filename))

        if check_loaded_options:
            # Ensure that for options defined in both the current model
            # options and the loaded model options, the values match between
            # current model and loaded model.
            loaded_options = data.get('options', {})
            current_options = vars(self.options)

            for option_name in \
                    (set(loaded_options.keys()) & set(current_options.keys())):
                if loaded_options[option_name] != current_options[option_name]:
                    raise ValueError(
                        f'Discrepancy between current and loaded model '
                        f'parameter: {option_name} '
                        f'loaded: {loaded_options[option_name]}, '
                        f'current: {current_options[option_name]}'
                    )

    def load_from(self, model):
        ''' Load from an existing model. State is not deep copied.
        To deep copy the model, uss ``clone``.
        '''
        if hasattr(model, 'option_map'):
            self.option_map = model.option_map

        if hasattr(model, 'params'):
            self.params = deepcopy(model.params)

        self.load_state_dict(model.state_dict())
        self.step = model.step

    def inc_step(self):
        ''' increment the step.
        ``step`` records the number of times the weight has been updated.'''
        self.step += 1

    def signature(self):
        '''Get model's signature.

        Returns:
            the model's signature string, specified by step.
        '''
        return "Model[%d]" % self.step

    def prepare_cooldown(self):
        """Prepare for "cooldown" forward passes (useful for batchnorm)."""
        pass


class Block(Model):

    def __init__(self, option_map, params):
        super().__init__(option_map, params)
        self.options = option_map
        self.relu = nn.LeakyReLU(0.1) if self.options.leaky_relu else nn.ReLU()
        self.conv_lower = self._conv_layer()
        self.conv_upper = self._conv_layer(relu=False)

    def _conv_layer(
            self,
            input_channel=None,
            output_channel=None,
            kernel=3,
            relu=True):
        if input_channel is None:
            input_channel = self.options.dim
        if output_channel is None:
            output_channel = self.options.dim

        layers = []
        layers.append(nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            padding=(kernel // 2),
        ))
        if self.options.bn:
            layers.append(
                nn.BatchNorm2d(output_channel,
                               momentum=(self.options.bn_momentum or None),
                               eps=self.options.bn_eps))
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def forward(self, s):
        s1 = self.conv_lower(s)
        s1 = self.conv_upper(s1)
        s1 = s1 + s
        s = self.relu(s1)
        return s


class GoResNet(Model):

    def __init__(self, option_map, params):
        super().__init__(option_map, params)
        self.options = option_map
        self.blocks = []
        for _ in range(self.options.num_block):
            self.blocks.append(Block(option_map, params))
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, s):
        return self.resnet(s)


class Model_PolicyValue(Model):

    def __init__(self, option_map, params):
        super().__init__(option_map, params)

        self.options = option_map
        self.board_size = params["board_size"]
        self.num_planes = params["num_planes"]
        # print("#future_action: " + str(self.num_future_actions))
        # print("#num_planes: " + str(self.num_planes))

        # Network structure of AlphaGo Zero
        # https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html

        # Simple method. multiple conv layers.
        self.relu = nn.LeakyReLU(0.1) if self.options.leaky_relu else nn.ReLU()
        last_planes = self.num_planes

        self.init_conv = self._conv_layer(last_planes)

        self.pi_final_conv = self._conv_layer(self.options.dim, 2, 1)
        self.value_final_conv = self._conv_layer(self.options.dim, 1, 1)

        d = self.board_size ** 2

        # Plus 1 for pass.
        self.pi_linear = nn.Linear(d * 2, d + 1)
        self.value_linear1 = nn.Linear(d, 256)
        self.value_linear2 = nn.Linear(256, 1)

        # Softmax as the final layer
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()
        self.resnet = GoResNet(option_map, params)

        if torch.cuda.is_available() and self.options.gpu is not None:
            self.init_conv.cuda(self.options.gpu)
            self.resnet.cuda(self.options.gpu)

        if self.options.use_data_parallel:
            if self.options.gpu is not None:
                self.init_conv = nn.DataParallel(
                    self.init_conv, output_device=self.options.gpu)
                self.resnet = nn.DataParallel(
                    self.resnet, output_device=self.options.gpu)

        self._check_and_init_distributed_model()

    def _check_and_init_distributed_model(self):
        if not self.options.use_data_parallel_distributed:
            return

        if not dist.is_initialized():
            world_size = self.options.dist_world_size
            url = self.options.dist_url
            rank = self.options.dist_rank
            # This is for SLURM's special use case
            if rank == -1:
                rank = int(os.environ.get("SLURM_NODEID"))

            print("=> Distributed training: world size: {}, rank: {}, URL: {}".
                  format(world_size, rank, url))

            dist.init_process_group(backend="nccl",
                                    init_method=url,
                                    rank=rank,
                                    world_size=world_size)

        # Initialize the distributed data parallel model
        master_gpu = self.options.gpu
        if master_gpu is None or master_gpu < 0:
            raise RuntimeError("Distributed training requires "
                               "to put the model on the GPU, but the GPU is "
                               "not given in the argument")
        # This is needed for distributed model since the distributed model
        # initialization will require the model be on the GPU, even though
        # the later code will put the same model on the GPU again with
        # self.options.gpu, so this should be ok
        # self.resnet.cuda(master_gpu)
        self.init_conv = nn.parallel.DistributedDataParallel(
            self.init_conv)
        self.resnet = nn.parallel.DistributedDataParallel(
            self.resnet)

    def _conv_layer(
            self,
            input_channel=None,
            output_channel=None,
            kernel=3,
            relu=True):
        if input_channel is None:
            input_channel = self.options.dim
        if output_channel is None:
            output_channel = self.options.dim

        layers = []
        layers.append(nn.Conv2d(
            input_channel,
            output_channel,
            kernel,
            padding=(kernel // 2)
        ))
        if self.options.bn:
            layers.append(
                nn.BatchNorm2d(output_channel,
                               momentum=(self.options.bn_momentum or None),
                               eps=self.options.bn_eps))
        if relu:
            layers.append(self.relu)

        return nn.Sequential(*layers)

    def prepare_cooldown(self):
        try:
            for module in self.modules():
                if module.__class__.__name__.startswith('BatchNorm'):
                    module.reset_running_stats()
        except Exception as e:
            print(e)
            print("The module doesn't have method 'reset_running_stats', "
                  "skipping. Please set bn_momentum to 0.1"
                  "(for cooldown = 50) in this case")

    def forward(self, x):
        s = x

        s = self.init_conv(s)
        s = self.resnet(s)

        d = self.board_size ** 2

        pi = self.pi_final_conv(s)
        pi = self.pi_linear(pi.view(-1, d * 2))
        logpi = self.logsoftmax(pi)
        pi = logpi.exp()

        V = self.value_final_conv(s)
        V = self.relu(self.value_linear1(V.view(-1, d)))
        V = self.value_linear2(V)
        V = self.tanh(V)

        return dict(logpi=logpi, pi=pi, V=V)

    def forward_to_resnet(self, x):
        s = x

        s = self.init_conv(s)
        s = self.resnet(s)

        return [s.detach()]

class DefaultModelOptions:
    leaky_relu = False
    dim = 256
    bn = True
    bn_momentum = 0.1
    bn_eps = 1e-5
    num_block = 20
    gpu = 0
    use_data_parallel = False
    use_data_parallel_distributed = False

params = {"board_size": 19, "num_planes":18}

def load_elf_model(path):
    model = Model_PolicyValue(DefaultModelOptions, params)
    _replace_prefix = ["resnet.module,resnet", "init_conv.module,init_conv"]
    replace_prefix = [
        item.split(",")
        for item in _replace_prefix
    ]
    model.load(
        path,
        omit_keys=[],
        replace_prefix=replace_prefix,
        check_loaded_options=False)
