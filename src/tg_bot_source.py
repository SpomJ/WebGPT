from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
import torch
import telebot
from web import *
from model import *
from duckduckgo_search import DDGS

import os
from pathlib import Path

# pth = os.path.abspath('Downloads/checkpoint-79000/content/drive/MyDrive/gpt-oasst/checkpoint-79000')

model_id = "facebook/opt-1.3b"

ADAPTER_PATH = "/home/user1/model-1/checkpoint-1500"
BOT_TOKEN = "6449634010:AAFMpPNmy1NEyxa45oVfjSsY_D1fDZxgQmo"
# HIST_CLEAR = '/empty'

model = Model(model_id, adapter=ADAPTER_PATH)


# bert = Model("bert-large-uncased-whole-word-masking-finetuned-squad")
class A:
    pass


bert = A()
bert.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", device="cuda:0")
bert.model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(
    'cuda')

bot = telebot.TeleBot(BOT_TOKEN)


def split_text(text, max_segment_length):
    segments = []
    current_segment = ""
    for paragraph in text.split("\\n"):  # Split by paragraphs for simplicity
        if len(current_segment) + len(paragraph) < max_segment_length:
            current_segment += paragraph + "\n"
        else:
            segments.append(current_segment.strip())
            current_segment = paragraph + "\n"
    if current_segment:  # Add the last segment
        segments.append(current_segment.strip())

    print(*map(lambda x: x.replace('\\n', ''), segments), sep='\n\n')
    return segments


def st2(text, max_segment_length):
    i = max_segment_length
    o = []
    while i < len(text):
        o.append(text[i - max_segment_length:i])
        i += max_segment_length // 2
    o.append(text[-max_segment_length:])
    return o


def bert_parse(text, question):
    max_segment_length = 512  # Maximum input length for BERT
    text_segments = st2(text, max_segment_length)
    # print(text_segments)
    best_answer = ""  # Initialize variables to store the best answer and its scores
    best_start_score = -float('inf')
    best_end_score = -float('inf')
    anss = []

    for i, segment in enumerate(text_segments):
        inputs = bert.tokenizer(question, segment, return_tensors="pt", truncation=True, max_length=512,
                                padding="max_length").to('cuda')

        with torch.no_grad():
            outputs = bert.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely answer
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        aes = torch.max(start_scores)
        es = torch.max(end_scores)

        if aes > best_start_score and es > best_end_score:
            best_start_score = aes
            best_end_score = es
            best_answer = bert.tokenizer.convert_tokens_to_string(
                bert.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            # print(best_answer)
            if '[SEP]' in best_answer: best_answer = ''
            anss.append(best_answer.replace('[PAD]', '').replace('[CLS]', ''))

    print(anss)
    return '; '.join(filter(bool, anss))

def results(q):
    texts = search_full(q, n=2)
    out = []
    for text in texts:
        out.append(bert_parse(text, q))
    print(out)
    return '\n'.join(filter(bool, out))

def respond(q):
    o = results(q)
    print(f'I1: {o}')
#    o = bert_parse(o, q)
#    print(f'I2: {o}')
    hist = [{'role': 'user', 'content': q},
            {'role': 'web', 'content': o}]
    input_text = Model.fmt(hist) + ROLE_TOKENS['bot']
    print(input_text)
    output = model.str_response(input_text)
    return output


@bot.message_handler(func=lambda x: 1)
def echo_all(message):
    question = message.text
    print(message.text)
    rply = respond(question)
    bot.reply_to(message, rply)


print('ready')
bot.infinity_polling()
