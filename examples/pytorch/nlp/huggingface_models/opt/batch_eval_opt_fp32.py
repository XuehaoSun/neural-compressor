import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
import sys
from transformers import set_seed
from torch.nn.functional import pad

sys.path.append('./')

set_seed(42)
import intel_extension_for_pytorch


class Evaluator:
    def __init__(self, dataset, tokenizer, device, batch_size=16):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            ##example = self.tokenizer(examples['text'], padding='max_length', max_length=195)
            example = self.tokenizer(examples['text'])
            return example

        self.dataloader = INCDataloader(dataset, tokenizer,batch_size, device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        index = 1
        for input_ids, label, label_indices in tqdm(self.dataloader):
            if index==1:
                model = torch.jit.trace(model, input_ids)
            outputs = model(input_ids)

            last_token_logits = outputs[0][:, label_indices, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if index % 100 == 0:
                print(hit / total)
            index += 1

        acc = hit / total
        return acc

    # @torch.no_grad()
    # def evaluate(self, model):
    #     model.eval()
    #     # The task is to predict the last word of the input.
    #     total, hit = 0, 0
    #     index = 1
    #     for batch in tqdm(self.dataset):
    #         input_ids = batch['input_ids'].unsqueeze(0)
    #         label = input_ids[:, -1]
    #         outputs = model(input_ids)
    #
    #         last_token_logits = outputs[0][:, -2, :]
    #         pred = last_token_logits.argmax(dim=-1)
    #         total += label.size(0)
    #         hit += (pred == label).sum().item()
    #         if index % 100 == 0:
    #             print(hit / total)
    #         index += 1
    #
    #     acc = hit / total
    #     return acc


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
        input_id = pad(input_id, (0, pad_len), value=1)

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


##model_name = "/data2/models/opt-125m/"
model_name = "facebook/opt-125m"
##model_name = "facebook/opt-6.7b"
model_name = "/mnt/datadrive/models_wenhuach/opt-66b/"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# dataset = load_dataset('lambada', split='validation')
# dataset = dataset.shuffle(seed=42)
# calib_dataloader = CalibDataloader(dataset, tokenizer, 'cpu')

dataset_eval = load_dataset('lambada', split='validation')
dataset_eval = dataset_eval.shuffle(seed=42)
evaluator = Evaluator(dataset_eval, tokenizer, 'cpu',batch_size=8)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
)
model.eval()


# tokenize the dataset
def tokenize_function(examples):
    global tokenizer
    example = tokenizer(examples['text'])
    return example


def eval_func(model):
    acc = evaluator.evaluate(model)
    return acc


acc = eval_func(model)
print(acc)
