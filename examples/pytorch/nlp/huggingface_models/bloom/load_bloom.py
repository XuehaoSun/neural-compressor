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
            # with torch.autograd.profiler.profile() as prof:
            #     outputs = model(input_ids)
            # table_res = prof.key_averages().table(sort_by="self_cpu_time_total")
            # print(table_res)
            last_token_logits = outputs[0][:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (index+1) % 10 == 0:
                print('#############')
                print('Temporary Accu:', hit / total, flush=True)
                print('#############')
        acc = hit / total
        return acc


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

evaluator = Evaluator(dataset, tokenizer, 8, 'cpu')

load_int8 = False
if load_int8:
    from neural_compressor.utils.pytorch import load
    model = load('saved_results')
else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
       model_name,
       torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
    )

acc = evaluator.evaluate(model)
print('######################')
print("Final Accu:", acc)
print('######################')
