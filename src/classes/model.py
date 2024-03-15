from trl import SFTTrainer
from ransformers import AutoTokenizer, AutoModelForCausalLM

ROLE_TOKENS = {
    'system': '[!]',
    'user':   '[>]',
    'web':    '[?]',
    'bot':    '[<]'
}

TOKENS = {
    'eos_token': '[/]',
    'pad_token': '[_]'
}

class Model:
    def __init__(model):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    @staticmethod
    def fmt(chain):
        '''
        Format the given chain into a string.
            chain: message chain in form of [{'role': '', 'content': ''}, ...]
        '''
        c = ''
        for msg in chain:
            c += ROLE_TOKEN[msg['role']]
            c += msg['content']
            c += S_END
        return c

#    def rev_fmt(self, tokens):
#        '''
#        Split string into a message chain in form of [{'role': '', 'content': ''}, ...]
#            tokens: String
#        '''
#        reverse_roles = dict().from
#        msgs = []
#        start = 0
#        token_sep = self.tokenizer(TOKENS['eos_token'])[0]
#        
#        for i in range(len(tokens)):
#            if tokens[i] == token_sep:
#                msgs.append({'role': role,  

class Trainer(Model):
    def __init__(self, dataset_paths, model, train_args={}, seq_len=512):
        if type(model) == str:
            model = AutoModelForCausalLM.from_pretrained(model)

        self.train_args = train_args
        self.dataset_paths = dataset_paths
        self.seq_len
        self.model = model
        self.dataset = load_dataset('json', data_files=dataset_paths, split='train')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.add_special_tokens({**TOKENS, 'additional_special_tokens': list(ROLE_TOKENS.values())})
        self.model.resize_token_embeddings(len(tokenizer))
        
    @staticmethod
    def bulk_fmt(dataset):
        '''
        Formats message chains from given dataset into strings.
        Returns list of said strings.
            dataset: Dataset in form of {"id": [0, 1, 2, ...], "messages": [[{"role": x, "content": x}, ...], ...]}
        '''
        O = []
        for chain in ds['messages']:
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


            

