import torch
import transformers
import sys
import time
from tqdm import tqdm
from datasets import load_dataset
import intel_extension_for_pytorch
import os
from torch.nn.functional import pad

sys.path.append('./')


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, device='cpu'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = INCDataloader(dataset, tokenizer, batch_size, device)
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        #last_index = 149
        last_index = -1
        acc_tmp = None
        num_samples_tmp = (last_index+1)*self.batch_size
        num_samples = 0
        for index, (batched_input, label, batched_index) in enumerate(tqdm(self.dataloader)):
            if index <= last_index:
                continue
            if index == last_index + 1:
                acc_tmp = 0.7220833333333333
            input_ids = batched_input
            start = time.time()
            outputs = model(input_ids)
            print()
            print('Time usage: ', time.time()-start)
            # with torch.autograd.profiler.profile() as prof:
            #     outputs = model(input_ids)
            # table_res = prof.key_averages().table(sort_by="self_cpu_time_total")
            # print(table_res)
            last_token_logits = outputs[0][:, batched_index, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            num_samples += len(label)
            if (index+1) % 10 == 0:
                if acc_tmp:
                    accu = ((hit / total)*num_samples + acc_tmp*num_samples_tmp)/(num_samples+num_samples_tmp)
                else:
                    accu = hit / total
                print('#############')
                print('Temporary Accu:', accu, flush=True)
                print('#############')
        if acc_tmp:
            accu = ((hit / total)*num_samples + acc_tmp*num_samples_tmp)/(num_samples+num_samples_tmp)
        else:
            accu = hit / total
        return accu

class INCDataloader():
    def __init__(self, dataset, tokenizer, batch_size=1, device='cpu'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        import math
        self.length = math.ceil(len(dataset) / self.batch_size)
        self.pad_len = 196

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def pad_input(self, input):
        input_id = input['input_ids'].unsqueeze(0)
        label = input_id[:, -1].to(self.device)
        pad_len = self.pad_len - input_id.shape[1]
        label_index = -2 - pad_len
        input_id = pad(input_id, (0, pad_len), value=3)

        return (input_id, label, label_index)

    def __iter__(self):
        input_ids = None
        labels = None
        label_indices = None
        for idx, batch in enumerate(self.dataset):
            input_id, label, label_index = self.pad_input(batch)

            if input_ids is None:
                input_ids = input_id
                labels = label
                label_indices = [label_index]
            else:
                input_ids = torch.cat((input_ids, input_id), 0)
                labels = torch.cat((labels, label), 0)
                label_indices.append(label_index)

            if (idx + 1) % self.batch_size == 0:
                yield (input_ids, labels, label_indices)
                input_ids = None
                labels = None
                label_indices = None
        if (idx + 1) % self.batch_size != 0:
            yield (input_ids, labels, label_indices)

    def __len__(self):
        return self.length


model_name = "/data2/dataset/bloom"
#model_name = "/dev/shm/bloom"
#model_name = "/data2/models/bloom-560m"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('lambada', split='validation')
dataset = dataset.shuffle(seed=42)
#dataset = dataset.select(range(12))

evaluator = Evaluator(dataset, tokenizer, 32, 'cpu')

load_int8 = False
if load_int8:
    from neural_compressor.utils.pytorch import load
    model = load("/data2/models/bloom-int8-no-sq")
else:
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.torchscript=True
    config.device_map='auto'
    config.torch_dtype='auto'
    model = transformers.AutoModelForCausalLM.from_config(config)

    from safetensors import safe_open
    for file in os.listdir(model_name):
        if not file.endswith('safetensors'):
            continue
        print(file)
        weights = {}
        with safe_open(os.path.join(model_name, file), framework='pt', device='cpu') as f:
            for k in f.keys():
                weights['transformer.'+k] = f.get_tensor(k)
        model.load_state_dict(weights, strict=False)
    model.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset('lambada', split='validation')
    input = tokenizer(dataset[0]['text'], padding='max_length', max_length=196, return_tensors='pt')
    example_input = input['input_ids'][0].to('cpu').unsqueeze(0)
    model = torch.jit.trace(model, example_input)
    model = torch.jit.freeze(model.eval())

acc = evaluator.evaluate(model)
print('######################')
print("Final Accu:", acc)
print('######################')
