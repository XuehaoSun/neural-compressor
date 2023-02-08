import torch

from tqdm import tqdm


class SmoothQuant:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dataset = dataloader
        self.input_maxs = {}
        self.calib_cnt = 100

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
            if name not in self.input_maxs.keys():
                self.input_maxs[name] = []
            input = inputs[0]
            ##TODO check input channel is correct
            if len(model.weight.shape) == 4:  ##conv3d or conv1d not suppoted now, need better way
                input = input.permute(0, 2, 3, 1)
            input = input.reshape(-1, input.shape[-1])
            max_tensor = torch.max(input, dim=0)[0]
            self.input_maxs[name].append(max_tensor)

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

    def calibration(self):
        norm_to_layer_mapping = self.get_layer_mapping()
        layer_to_norm_mapping = {}
        for key in norm_to_layer_mapping:
            for layer_name in norm_to_layer_mapping[key]:
                layer_to_norm_mapping[layer_name] = key
        hook_module_names_tmp = [norm_to_layer_mapping[key][0] for key in norm_to_layer_mapping.keys()]
        hook_modules = {}

        for index, name in enumerate(hook_module_names_tmp):
            module = self.get_module(name)
            if isinstance(module, torch.nn.Linear) or isinstance(module,
                                                                 torch.nn.Conv2d):  ##TODO remove group conv later
                hook_modules[name] = module
        if len(hook_modules) == 0:
            return self.model

        self.add_observer(hook_modules)
        self.dump_min_max()
        self.remove_observer()
        return norm_to_layer_mapping


    def dump_min_max(self, calibration_method="min_max"):
        cnt = 1
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)

            label = input_ids[:, -1]
            attention_mask = torch.ones_like(input_ids)
            self.model(input_ids, attention_mask=attention_mask)
            cnt += len(batch)
            if cnt > self.calib_cnt:
                break
        ##stack
        for key in self.input_maxs.keys():
            val = self.input_maxs[key]
            val = torch.stack(val, dim=0)
            val = torch.max(torch.abs(val), dim=0)[0]  ##FIXME should add abs
            self.input_maxs[key] = val

    def adjust_parameters(self, norm_to_layer_mapping, input_maxs, alpha=0.5):
        norm_to_input_maxs = {}
        for key in norm_to_layer_mapping.keys():
            layer_name = norm_to_layer_mapping[key][0]
            norm_to_input_maxs[key] = input_maxs[layer_name]

        for index, key in enumerate(norm_to_layer_mapping.keys()):
            input_max = norm_to_input_maxs[key]
            layers = norm_to_layer_mapping[key]
            weights = []
            for layer in layers:
                weight = self.get_module(layer).weight  ##TODO oc*ic, support transposed conv
                if len(weight.shape) == 4:
                    weight = weight.permute(0, 2, 3, 1)
                    weight = weight.reshpae(-1, weight.shape[-1])
                weights.append(weight)

            weights = torch.cat(weights, dim=0)

            weight_max_per_channel = torch.max(torch.abs(weights), dim=0)[0]  ##FIXME abs
            input_power = torch.pow(input_max, alpha)
            print(max(input_max), min(input_power))
            weight_power = torch.pow(weight_max_per_channel, 1 - alpha)

            ##adjust parameters
            scale = torch.clip(input_power / weight_power, min=1e-5)

            norm_layer = self.get_module(key)
            weights_layers = []
            layers = norm_to_layer_mapping[key]

            for layer in layers:
                layer = self.get_module(layer)
                weights_layers.append(layer)
            ##TODO if norm does not have affine, need to add one
            norm_layer.weight /= scale
            norm_layer.bias /= scale
            for layer in weights_layers:
                layer.weight *= scale

    def transform(self, alpha):
        with torch.no_grad():
            hook_modules, layer_to_norm_mapping, norm_to_layer_mapping = self.calibration()
            self.adjust_parameters(norm_to_layer_mapping, self.input_maxs)
            return self.model

    def get_layer_mapping(self):
        tg = TorchGraphAnalysis()
        for batch in tqdm(self.dataset):
            example_inputs = batch['input_ids'].to(self.device).unsqueeze(0)
            break
        try:
            layer_mapping = tg.get_norm_to_layer_mapping(self.model, example_inputs)
        except:
            layer_mapping = {}##todo, if strict mode=False, we hook all the mat/conv layer insert mul layer
        del tg
        return layer_mapping


class TorchGraphAnalysis:
    def __init__(self):
        self.aten_to_op = {
            "aten::linear": "linear",  ##TODO support conv
            "aten::layer_norm": "layer_norm",
            "aten::to": "to",
        }
        self.skip_layers_to_find_norm = ["aten::to",
                                         ##"aten::size",
                                         ##"prim::NumToTensor"
                                         ]  ##TODO, current skip layer may be incomplete, matmul/conv->Relu/leakyRelu->Matmul/conv is an option
        self.skip_layers_to_find_norm = ["aten::to"]
        self.norm_layers = ["layer_norm"]  ##TODO,suppport more norm

    def trace(self, model, dummy_input):
        traced_model = torch.jit.trace(model, dummy_input, strict=False)  ##TODO add a try catch
        traced_model = torch.jit.freeze(traced_model.eval())
        self.traced_model = traced_model  ##TODO,need to check memory usage

    def kind_to_op_type(self, kind):
        if kind in self.aten_to_op:
            return self.aten_to_op[kind]
        else:
            return kind.split("::")[-1]

    def op_type_to_kind(self, op_type):
        for key in self.aten_to_op.keys():
            if op_type == self.aten_to_op[key]:
                return key

        return "aten::" + str(op_type)  ##not correct

    def get_parent(self, node):
        if node.inputs() == None:
            return None
        return list(node.inputs())[0].node()

    def get_nodes(self, op_types=['linear']):
        if isinstance(op_types, str):
            op_types = [op_types]
        nodes = []
        for node in self.traced_model.graph.nodes():
            node_type = node.kind()
            for op_type in op_types:
                if self.kind_to_op_type(node_type) == op_type:
                    nodes.append((node, op_type))
                    break
        return nodes

    def get_prev_norm_layer(self, nodes):  ## TODO may be not correct for other models
        norm_mapping = []
        for node in nodes:
            parent = self.get_parent(node)
            while 1:
                if parent.kind() in self.skip_layers_to_find_norm:
                    parent = self.get_parent(parent)
                    continue
                if self.kind_to_op_type(parent.kind()) in self.norm_layers:
                    norm_mapping.append(parent)
                else:
                    norm_mapping.append(None)
                break
        return norm_mapping

    def get_norm_to_layer_mapping(self, model, example_input):
        ##self.model = model
        self.trace(model, example_input)
        nodes_types = self.get_nodes(["linear"])
        nodes = [node_type[0] for node_type in nodes_types]
        nodes_prev_norm = self.get_prev_norm_layer(nodes)
        norm_to_layer_mapping = {}
        for index, norm in enumerate(nodes_prev_norm):
            if norm == None:
                continue
            node = nodes[index]
            layer_name = '.'.join(node.scopeName().split('/')[-1].split('.')[1:])
            norm_name = '.'.join(norm.scopeName().split('/')[-1].split('.')[1:])
            if norm_name in norm_to_layer_mapping.keys():
                norm_to_layer_mapping[norm_name].append(layer_name)
            else:
                norm_to_layer_mapping[norm_name] = [layer_name]
        del self.traced_model
        return norm_to_layer_mapping