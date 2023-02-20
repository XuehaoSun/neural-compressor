import torch
import transformers
import sys
import time
from tqdm import tqdm
from datasets import load_dataset

sys.path.append('./')


class SmoothQuant:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.dataloader = dataloader
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

    ##https://gist.github.com/sailfish009/28b54c8aa6398148a6358b8f03c0b611
    def percentile(t: torch.tensor, q: float):
        """
        Return the ``q``-th percentile of the flattened input tensor's data.

        CAUTION:
         * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
         * Values are not interpolated, which corresponds to
           ``numpy.percentile(..., interpolation="nearest")``.

        :param t: Input tensor.
        :param q: Percentile to compute, which must be between 0 and 100 inclusive.
        :return: Resulting value (scalar).
        """
        # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
        # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
        # so that ``round()`` returns an integer, even if q is a np.float32.
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

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
        for batch in tqdm(self.dataloader):
            input_ids = batch
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

    def load(self):
        with torch.no_grad():
            state_dict = torch.load('sq_state_dict.bin')
            norm_to_layer_mapping = state_dict['norm_to_layer_mapping']
            self.input_maxs = state_dict['input_maxs']
            self.adjust_parameters(norm_to_layer_mapping, self.input_maxs)
            return self.model

    def transform(self):
        with torch.no_grad():
            norm_to_layer_mapping = self.calibration()
            state_dict = {
                'norm_to_layer_mapping': norm_to_layer_mapping,
                'input_maxs': self.input_maxs,
            }
            torch.save(state_dict, 'sq_state_dict.bin')
            self.adjust_parameters(norm_to_layer_mapping, self.input_maxs)
            return self.model

    def get_layer_mapping(self):
        tg = TorchGraphAnalysis()
        for batch in tqdm(self.dataloader):
            example_inputs = batch
            break

        layer_mapping = tg.get_norm_to_layer_mapping(self.model, example_inputs)
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

class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, device='cpu'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        # the max_length of validation dataset is 195
        self.dataloader = INCDataloader(dataset, tokenizer, batch_size, device, max_length=195)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for index, batch in enumerate(tqdm(self.dataloader)):
            input_ids = batch
            label = input_ids[:, -1]
            start = time.time()
            outputs = model(input_ids)
            print()
            print('Time usage: ', time.time()-start)
            last_token_logits = outputs[0][:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (index+1) % 10 == 0:
                print()
                print('Tmp Accu:', hit / total, flush=True)
        acc = hit / total
        return acc


class INCDataloader():
    def __init__(self, dataset, tokenizer, batch_size=1, device='cpu', max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        import math
        self.length = math.ceil(len(dataset) / self.batch_size)

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'], padding='max_length', max_length=max_length)
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def __iter__(self):
        batched_input = None
        for idx, batch in enumerate(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]
            if batched_input is None:
                batched_input = input_ids
            else:
                batched_input = torch.cat((batched_input, input_ids), 0)
            if (idx + 1) % self.batch_size == 0:
                yield batched_input
                batched_input = None
        if (idx + 1) % self.batch_size != 0:
            yield batched_input
    
    def __len__(self):
        return self.length


model_name = "bigscience/bloom-560m"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('lambada', split='validation')
dataset = dataset.shuffle(seed=42)
train_dataset = load_dataset('lambada', split='train')
train_dataset = train_dataset.shuffle(seed=42)

calib_dataset = train_dataset.select(range(100))
calib_dataloader = INCDataloader(calib_dataset, tokenizer, 16, 'cpu')

evaluator = Evaluator(dataset, tokenizer, 16, 'cpu')

model = transformers.AutoModelForCausalLM.from_pretrained(
   model_name,
   torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
)

# Enable smoothquant to transfer weights before quantization
model.eval()
sq = SmoothQuant(model, calib_dataloader)
model = sq.transform()
# if transform is executed before, you can use load to avoid duplicated calibration.
# model = sq.load()

def eval_func(model):
    acc = evaluator.evaluate(model)
    return acc


from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization

conf = PostTrainingQuantConfig(backend='ipex', excluded_precisions=["bf16"])
conf.performance_only=True
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=calib_dataloader,
                           eval_func=eval_func)

q_model.save('saved_results')
