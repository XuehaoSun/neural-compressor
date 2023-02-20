import torch
import transformers
import sys
import time
from tqdm import tqdm
from datasets import load_dataset
import intel_extension_for_pytorch

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
        last_index = 136
        last_index = -1
        acc_tmp = None
        num_samples_tmp = (last_index+1)*self.batch_size
        num_samples = 0
        for index, batch in enumerate(tqdm(self.dataloader)):
            if index <= last_index:
                continue
            if index == last_index + 1:
                acc_tmp = 0.4607664233576642
            input_ids = batch
            label = input_ids[:, -1]
            start = time.time()
            outputs = model(input_ids)
            print()
            print('Time usage: ', time.time()-start)
            # with torch.autograd.profiler.profile() as prof:
            #     outputs = model(input_ids)
            # table_res = prof.key_averages().table(sort_by="self_cpu_time_total")
            # print(table_res)
            last_token_logits = outputs[0][:, -2, :]
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

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'], padding='max_length', max_length=195)
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def __iter__(self):
        batched_input = None
        for idx, batch in enumerate(self.dataset):
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
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

evaluator = Evaluator(dataset, tokenizer, 16, 'cpu')

load_int8 = False
if load_int8:
    from neural_compressor.utils.pytorch import load
    model = load('saved_results')
else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
       model_name,
       torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset('lambada', split='validation')
    input = tokenizer( dataset[0]['text'], padding='max_length', max_length=195, return_tensors='pt')
    example_input = input['input_ids'][0].to('cpu').unsqueeze(0)
    model = torch.jit.trace(model, example_input)

acc = evaluator.evaluate(model)
print('######################')
print("Final Accu:", acc)
print('######################')
