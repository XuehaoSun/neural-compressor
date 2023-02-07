import torch
import transformers
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
# from smoothquant.smooth import smooth_lm
import torch
from tqdm import tqdm
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('./')





class TorchGraphAnalysis:
    def __init__(self):
        self.aten_to_op = {
            "aten::linear": "linear",  ##TODO support conv
            "aten::layer_norm": "layer_norm",
            "aten::to": "to",
        }
        self.skip_layers_to_find_norm = ["aten::to", "aten::size",
                                         "prim::NumToTensor"]  ##TODO, current skip layer may be incomplete
        self.skip_layers_to_find_norm = ["aten::to"]
        self.norm_layers = ["layer_norm"]  ##TODO,suppport more norm

    def trace(self, model, dummy_input):
        traced_model = torch.jit.trace(model, dummy_input, strict=False)##TODO add a try catch
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
        self.model = model
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

    #
    # def get_orig_module_name(self, node):
    #     node.scope



class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])
        self.trace_analysis = TorchGraphAnalysis()

    def trace_model(self, model):
        model.eval()
        # The task is to predict the last word of the input.

        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            attention_mask = torch.ones_like(input_ids)

            example_input = [input_ids]
            self.trace_analysis = TorchGraphAnalysis()
            norm_to_layer_mapping = self.trace_analysis.get_norm_to_layer_mapping(model, example_input)



    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "logits"):
                last_token_logits = outputs.logits[:, -2, :]
            else:
                last_token_logits= outputs[0][:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc


from datasets import load_dataset

model_name = "bigscience/bloom-560m"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset('lambada', split='validation',
                       ##download_mode="force_redownload",
                       )

evaluator = Evaluator(dataset, tokenizer, 'cuda')

model_fp16 = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                               ##torch_dtype=torch.float16,
                                                               device_map='auto',
                                                               torchscript=True##TODO, if false, trace will fail
                                                               )

# evaluator.trace_model(model_fp16)

acc_fp16 = evaluator.evaluate(model_fp16)
print(f'Original model (fp16) accuracy: {acc_fp16}')
