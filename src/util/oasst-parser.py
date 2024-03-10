#!/usr/bin/python3

import json

def walk(d, h=[], is_user=1):
    if not d['replies']:
        if is_user:
            return [h]
        return [h + [parse(d, is_user)]]
    o = []
    for r in d['replies']:
        o += walk(r, h + [parse(d, is_user)], not is_user)
    return o


def parse(d, is_user):
    return {
        'role': 'user' if is_user else 'bot',
        'content': d['text']
    }
        
        
chains = []
with open('./ds_raw.jsonl', 'r') as I:
    for l in I.readlines():
        d = json.loads(l)
        chains += walk(d['prompt'])

for i in range(len(chains)):
    if chains[i]:
    	print(json.dumps({'id': i, 'messages': chains[i]}))
