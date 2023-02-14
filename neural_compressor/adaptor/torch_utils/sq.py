import torch
from ...utils import logger


def model_forward(model, dataloader, sample_cnt):
    try:
        cnt = 0
        for idx, (input, label) in enumerate(dataloader):
            output = model(input)
            cnt += len(input)
            if cnt >= sample_cnt:
                break
    except Exception as e:
        cnt = 0
        for idx, input in enumerate(dataloader):
            if isinstance(input, dict):
                out = model(**input)
            else:
                out = model(input)
            cnt += len(input)
            if cnt >= sample_cnt:
                break


class SmoothQuant:
    def __init__(self, model: torch.nn.Module, dataloader, traced_model=None):
        self.model = model
        device, dtype = self.get_device()
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self.dataloader = dataloader
        self.input_maxes = {}
        self.traced_model = traced_model
        if self.traced_model == None:
            self.traced_model = self.model
        self.weight_scale_info = {}
        self.absorb_scales_info = {}

    def get_device(self):
        for _, p in self.model.named_parameters():
            return p.data.device, p.data.dtype

    def get_module(self, key):
        attrs = key.split('.')
        module = self.model
        for attr in attrs:
            try:
                attr = int(attr)
                module = module[attr]
            except:
                module = getattr(module, attr)
        return module

    def save_input_pc_hook(self, name):
        def save_input_hook(model, inputs, outputs):
            if name not in self.input_maxes.keys():
                self.input_maxes[name] = []
            input = inputs[0]
            ##TODO check input channel is correct
            if len(model.weight.shape) == 4:  ##conv3d or conv1d not suppoted now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            self.input_maxes[name].append(max_tensor)

        return save_input_hook

    def add_observer(self, modules):
        self.hook_handles = []
        for key in modules.keys():
            hook_func = self.save_input_pc_hook(key)
            hook_handle = modules[key].register_forward_hook(hook_func)
            self.hook_handles.append(hook_handle)

    def remove_observer(self):
        for hook_handle in self.hook_handles:
            hook_handle.remove()

    # ##https://gist.github.com/sailfish009/28b54c8aa6398148a6358b8f03c0b611
    # def percentile(t: torch.tensor, q: float):
    #     """
    #     Return the ``q``-th percentile of the flattened input tensor's data.
    #
    #     CAUTION:
    #      * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
    #      * Values are not interpolated, which corresponds to
    #        ``numpy.percentile(..., interpolation="nearest")``.
    #
    #     :param t: Input tensor.
    #     :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    #     :return: Resulting value (scalar).
    #     """
    #     # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    #     # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    #     # so that ``round()`` returns an integer, even if q is a np.float32.
    #     k = 1 + round(.01 * float(q) * (t.numel() - 1))
    #     result = t.view(-1).kthvalue(k).values.item()
    #     return result

    def calibration(self, absorb_to_layer, calib_num):
        layer_to_absorb = {}
        for key in absorb_to_layer:
            for layer_name in absorb_to_layer[key]:
                layer_to_absorb[layer_name] = key
        hook_module_names_tmp = [absorb_to_layer[key][0] for key in absorb_to_layer.keys()]
        hook_modules = {}

        for index, name in enumerate(hook_module_names_tmp):
            module = self.get_module(name)
            if isinstance(module, torch.nn.Linear) or isinstance(module,
                                                                 torch.nn.Conv2d):
                if isinstance(module, torch.nn.Conv2d):
                    if module.groups > 1 and module.in_channels == module.out_channels and module.groups == module.in_channels:
                        continue
                    else:
                        pass

                hook_modules[name] = module
        if len(hook_modules) == 0:
            return {}

        self.add_observer(hook_modules)
        self.dump_min_max(calib_num=calib_num)
        self.remove_observer()
        return self.input_maxes

    def dump_min_max(self, calibration_method="min_max", calib_num=100):
        model_forward(self.model, self.dataloader, calib_num)
        ##stack
        for key in self.input_maxes.keys():
            val = self.input_maxes[key]
            val = torch.stack(val, dim=0)
            val = torch.max(torch.abs(val), dim=0)[0]  ##FIXME should add abs
            self.input_maxes[key] = val

    def reshape_in_channel_to_last(self, layer_name):
        weight = self.get_module(layer_name).weight  ##TODO oc*ic, support transposed conv
        if len(weight.shape) == 4:
            weight = weight.permute(0, 2, 3, 1)
            weight = weight.reshape(-1, weight.shape[-1])
        return weight

    def scale_layer_weight(self, layer_name, scale: torch.Tensor):##input channel
        layer = self.get_module(layer_name)
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
            scale = scale.view(1, scale.shape[0], 1, 1)
            layer.weight *= scale
        elif isinstance(layer, torch.nn.Linear):
            scale = scale.view(1, scale.shape[0])
            layer.weight *= scale
        else:
            logger.warning(f"found unsupported layer {type(layer)}, try to multiply scale directly ")
            layer.weight *= scale

    def absorb_scales(self, layer_name, scale):##output channel
        layer = self.get_module(layer_name)
        if isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.GroupNorm) or isinstance(layer,
                                                                                                          torch.nn.InstanceNorm2d):
            if layer.affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(
                    weight, requires_grad=False)
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(bias, requires_grad=False
                                                )
        elif isinstance(layer, torch.nn.LayerNorm):
            if layer.elementwise_affine:
                layer.weight *= scale
                layer.bias *= scale
            else:
                layer.elementwise_affine = True
                weight = torch.ones(layer.num_features, device=self.device, dtype=self.dtype) * scale
                layer.weight = torch.nn.Parameter(
                    torch.ones(weight, requires_grad=False))
                bias = torch.zeros(layer.num_features, device=self.device, dtype=self.dtype)
                layer.bias = torch.nn.Parameter(
                    bias, requires_grad=False)

        elif isinstance(layer, torch.nn.Conv2d):
            ##the order could not be changed
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(-1, scale.shape[0], 1, 1)
            layer.weight *= scale
        elif isinstance(layer, torch.nn.Linear):
            if hasattr(layer, "bias") and (layer.bias != None):
                layer.bias *= scale
            scale = scale.view(scale.shape[0],1)
            layer.weight *= scale

        else:
            logger.warning(f"found unsupported layer {type(layer)}, try to multiply scale to weight and bias directly ")
            if hasattr(layer, "weight") and layer.weight != None:
                layer.weight *= scale
            if hasattr(layer, "bias") and layer.bias != None:
                layer.bias *= scale

    def adjust_parameters(self, absorb_to_layer, input_maxes, alpha=0.5):
        absorb_to_input_maxes = {}
        for key in absorb_to_layer.keys():
            layer_name = absorb_to_layer[key][0]
            absorb_to_input_maxes[key] = input_maxes[layer_name]

        weight_scales_info = {}
        absorb_scales_info = {}
        for index, key in enumerate(absorb_to_layer.keys()):
            input_max = absorb_to_input_maxes[key]
            layers = absorb_to_layer[key]
            weights = []
            for layer in layers:
                weight = self.reshape_in_channel_to_last(layer)
                weights.append(weight)

            weights = torch.cat(weights, dim=0)

            weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]
            input_power = torch.pow(input_max, alpha)
            logger.info(f"{max(input_max)}, {min(input_power)}")  ##TODO changed it to debug later
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha)

            ##adjust parameters
            scale = torch.clip(input_power / weight_power, min=1e-5)

            self.absorb_scales(key, 1.0 / scale)
            absorb_scales_info[key] = 1.0 / scale
            layer_names = absorb_to_layer[key]
            for layer_name in layer_names:
                self.scale_layer_weight(layer_name, scale)
                weight_scales_info[layer_name] = scale
        return weight_scales_info, absorb_scales_info

    def transform(self, alpha=0.5, percentile=99.999, op_types=['Linear', 'Conv2d', 'ConvTranspose2d'],
                  scales_per_op=False, calib_num=100):

        with torch.no_grad():
            absorb_to_layer, no_absorb_layers = self.trace(
                op_types)  ##TODO we need to insert mul layer for no_absorb_layers later
            if absorb_to_layer == None:
                logger.warning("sorry, could not trace the model")  ##TODO convert to insert mul mode
                return self.model

            input_maxes = self.calibration(absorb_to_layer, calib_num)

            self.weight_scale_info, self.absorb_scales_info = self.adjust_parameters(absorb_to_layer, input_maxes,
                                                                                     alpha)
            return self.model

    def recover(self):
        with torch.no_grad():
            for key in self.weight_scale_info:
                self.scale_layer_weight(key, 1.0 / self.weight_scale_info[key])
            for key in self.absorb_scales_info:
                self.absorb_scales(key, 1.0 / self.absorb_scales_info[key])

    def trace(self, op_types):
        tg = TorchGraphAnalysis()
        for idx, input in enumerate(self.dataloader):
            example_inputs = input
            break
        # for batch in (self.dataset):
        #     example_inputs = batch['input_ids'].to(self.device).unsqueeze(0)
        #     break

        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.traced_model, example_inputs, op_types)
        return absorb_to_layer, no_absorb_layers


class TorchGraphAnalysis:
    def __init__(self):
        # self.aten_to_op = {
        #     "aten::linear": "Linear",
        #     "aten::layer_norm": "layer_norm",
        #     "aten::to": "to",
        #     "aten::_convolution": "Conv",
        #     "aten::group_norm": "group_norm",
        #     "aten::batch_norm": "batch_norm",
        #     "aten::instance_norm": "instance_norm"
        # }

        self.supported_torch_module_to_aten = {
            "Linear": "aten::linear",
            "Conv2d": "aten::_convolution",
            "ConvTranspose2d": "aten::_convolution",
            "LayerNorm": "layer_norm",
            "BatchNorm2d": "",
            "GroupNorm": "aten::group_norm",
            "InstanceNorm2d": "instance_norm",
        }
        ##TODO, must statisfy ax=f(ax),current skip layer may be incomplete
        self.skip_ops_to_find_absorb = ["aten::to",
                                        "aten::relu",
                                        "aten::leaky_relu"
                                        ]

        self.could_absorb_layers = ["aten::layer_norm", "aten::batch_norm", "aten::linear", "aten::_convolution",
                                    "aten::group_norm",
                                    "aten::instance_norm"]  ##TODO,suppport more norm

    def trace(self, model, dummy_input):
        traced_model = None
        optimize_numerics = False
        if isinstance(dummy_input, dict):
            try:
                traced_model = torch.jit.trace(model, dummy_input["input_ids"], strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                pass
        else:
            try:
                traced_model = torch.jit.trace(model, dummy_input, strict=False)
                traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
            except:
                try:
                    traced_model = torch.jit.trace(model, dummy_input[0], strict=False)
                    traced_model = torch.jit.freeze(traced_model.eval(), optimize_numerics=optimize_numerics)
                except:
                    assert False, "can't trace the model"
        return traced_model

    # def kind_to_op_type(self, kind):
    #     if kind in self.aten_to_op:
    #         return self.aten_to_op[kind]
    #     else:
    #         return kind.split("::")[-1]

    def get_parent(self, node):
        if node.inputs() == None:
            return None
        return list(node.inputs())[0].node()

    def get_nodes(self, traced_model, op_types=['Linear']):
        if isinstance(op_types, str):
            op_types = [op_types]
        nodes = []
        for node in traced_model.graph.nodes():
            node_type = node.kind()
            ##print(node_type)
            for op_type in op_types:
                if node_type == op_type:
                    nodes.append((node, op_type))
                    break
        return nodes

    def get_prev_absorb_layer(self, nodes):
        prev_absorb_layer = []
        for node in nodes:
            parent = self.get_parent(node)
            while 1:
                if parent.kind() in self.skip_ops_to_find_absorb:
                    parent = self.get_parent(parent)
                    continue
                if parent.kind() in self.could_absorb_layers:
                    prev_absorb_layer.append(parent)
                else:
                    prev_absorb_layer.append(None)
                break
        return prev_absorb_layer

    def mapping_torch_module_to_aten(self, op_types):
        res = []
        for op in op_types:
            if op not in self.supported_torch_module_to_aten.keys():
                logger.warning(f"{op} is not supported in smooth quant, ignoring...")
                continue
            res.append(self.supported_torch_module_to_aten[op])
        res = list(set(res))
        return res

    def get_absorb_to_layer(self, model, example_input, op_types):
        traced_model = self.trace(model, example_input)
        if traced_model == None:
            return None
        aten_op_types = self.mapping_torch_module_to_aten(op_types)
        nodes_types = self.get_nodes(traced_model, aten_op_types)
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_absorb = self.get_prev_absorb_layer(nodes)
        absorb_to_layer = {}
        no_absorb_layers = []
        for index, absorb in enumerate(nodes_prev_absorb):
            if absorb == None:
                no_absorb_layers.append(nodes[index])
                continue
            node = nodes[index]
            layer_name = '.'.join(node.scopeName().split('/')[-1].split('.')[1:])
            absorb_name = '.'.join(absorb.scopeName().split('/')[-1].split('.')[1:])
            if absorb_name in absorb_to_layer.keys():
                absorb_to_layer[absorb_name].append(layer_name)
            else:
                absorb_to_layer[absorb_name] = [layer_name]
        absorb_to_layer = self.remove_unsupported_layers(model, absorb_to_layer)
        return absorb_to_layer, no_absorb_layers

    def remove_unsupported_layers(self, model, absorb_to_layer):
        res = {}

        for key in absorb_to_layer.keys():

            absorb_layer = self.get_module(model, key)
            layer_type = absorb_layer.__class__.__name__
            if layer_type not in self.supported_torch_module_to_aten.keys():
                continue
            supported = True
            for layer_name in absorb_to_layer[key]:
                layer = self.get_module(model, layer_name)
                layer_type = layer.__class__.__name__
                if layer_type not in self.supported_torch_module_to_aten.keys():
                    supported = False
                    break
            if supported:
                res[key] = absorb_to_layer[key]
        return res

    def get_module(self, model, key):
        attrs = key.split('.')
        module = model
        for attr in attrs:
            try:
                attr = int(attr)
                module = module[attr]
            except:
                module = getattr(module, attr)
        return module
