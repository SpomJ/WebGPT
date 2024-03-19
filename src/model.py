from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import pathlib

# ROLE_TOKENS = {
#     'system': '[!]',
#     'user':   '[>]',
#     'web':    '[?]',
#     'bot':    '[<]'
# }
ROLE_TOKENS = {
    'system': '[SYS]',
    'user':   '[USR]',
    'web':    '[WEB]',
    'bot':    '[BOT]'
}

TOKENS = {
    'eos_token': '[/]',
    'pad_token': '[_]'
}

class Model:
    def __init__(self, model, seq_len=512, use_gpu=1):
        # if type(model) in [str, pathlib.Path]:
        self.tokenizer = AutoTokenizer.from_pretrained(model, device='cuda:0' if use_gpu else 'cpu')
        self.model = model
        # if type(self.model) in [str, pathlib.WindowsPath]:
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.seq_len = seq_len
        self.use_gpu = use_gpu
        if use_gpu:
            self.model = self.model.to('cuda')

    @staticmethod
    def fmt(chain):
        '''
        Format the given chain into a string.
            chain: message chain in form of [{'role': '', 'content': ''}, ...]
        '''
        c = ''
        for msg in chain:
            c += ROLE_TOKENS[msg['role']]
            c += msg['content']
            c += TOKENS['eos_token']
        return c

#    def rev_fmt(self, tokens):
#        '''
#        Split token sequence into a message chain in form of [{'role': '', 'content': ''}, ...]
#            tokens: list[str]
#        '''
#        reverse_roles = dict().from
#        msgs = []
#        start = 0
#        token_sep = self.tokenizer(TOKENS['eos_token'])[0]
#        
#        for i in range(len(tokens)):
#            if tokens[i] == token_sep:
#                msgs.append({'role': role,

    def str_response(self, s):
        p = self.tokenizer(s)
        input_ids = self.tokenizer(s, return_tensors="pt").to('cuda')
        output = self.model.generate(
            **input_ids,
            max_length=512,
            repetition_penalty=10.,
            encoder_repetition_penalty=10.,
            eos_token_id=self.tokenizer(TOKENS['eos_token'])['input_ids'][0])
        return self.tokenizer.decode(output[0][len(p['input_ids']):])

class Trainer(Model):
    def __init__(self, dataset_paths, model, train_args={}, seq_len=512, use_gpu=True, lora_config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        if type(model) == str:
            if lora_config:
                model = AutoModelForCausalLMWithValueHead.from_pretrained(model, peft_config=lora_config)
            else:
            	model = AutoModelForCausalLM.from_pretrained(model)

        self.train_args = train_args
        self.dataset_paths = dataset_paths
        self.seq_len = seq_len
        self.model = model
        self.dataset = load_dataset('json', data_files=dataset_paths, split='train')
        self.tokenizer.add_special_tokens({**TOKENS, 'additional_special_tokens': list(ROLE_TOKENS.values())})
        self.model.resize_token_embeddings(len(self.tokenizer))
        if use_gpu:
            self.model = self.model.to('cuda')
        
    @classmethod
    def bulk_fmt(self, dataset):
        '''
        Formats message chains from given dataset into strings.
        Returns list of said strings.
            dataset: Dataset in form of {"id": [0, 1, 2, ...], "messages": [[{"role": x, "content": x}, ...], ...]}
        '''
        O = []
        for chain in dataset['messages']:
            O.append(self.fmt(chain))
        return O

    def test_tokenizer(self, s, output_token_nums=0):
        '''
        Test tokenizer by splitting an input string
            s:                 string to split
            output_token_nums: whether to return the split text or tokes
        '''
        t = self.tokenizer(s)
        if not output_token_nums:
            t = list(map(self.tokenizer.decode, t))
        return t

    def train(self, use_default_args=1, **cust_args):
        SFTTrainer(
            model =           self.model,
            train_dataset =   self.dataset,
            tokenizer =       self.tokenizer,
            formatting_func = self.bulk_fmt,
            max_seq_length =  self.seq_len,
            args =            TrainingArguments(**{**self.train_args, **cust_args})  # {**a, **b} <=> a.update(b), but not in-place
        ).train()

            

