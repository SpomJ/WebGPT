from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
import torch
import telebot
from web import *
from model import *
from duckduckgo_search import DDGS

import os
from pathlib import Path
# pth = os.path.abspath('Downloads/checkpoint-79000/content/drive/MyDrive/gpt-oasst/checkpoint-79000')

MODEL_PATH = Path(r"C:\Users\Sirius\Downloads\checkpoint-79000\content\drive\MyDrive\gpt-oasst\checkpoint-79000")
BOT_TOKEN = "6449634010:AAFMpPNmy1NEyxa45oVfjSsY_D1fDZxgQmo"
HIST_CLEAR = '/empty'

model = Model(MODEL_PATH, use_gpu=1)
print(type(model.model))

# bert = Model("bert-large-uncased-whole-word-masking-finetuned-squad")
class A:
    pass
bert = A()
bert.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", device="cuda:0")
bert.model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to('cuda')

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
        i += max_segment_length//2
    o.append(text[-max_segment_length:])
    return o


def results(question):
    text = search_full(question)
    if not text: text = ''
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
    return '; '.join(anss)


def respond(hist, q):
    hist = hist + [
        {'role': 'user', 'content': q},
        {'role': 'web', 'content': results(q)}]
    input_text = Model.fmt(hist)
    print(input_text)
    output = model.str_response(input_text)
    return hist, output


hist = []


@bot.message_handler(func=lambda x: 1)
def echo_all(message):
    global hist
    question = message.text
    print(message.text)
    print(hist)
    if message.text == HIST_CLEAR:
        hist = []
        bot.reply_to(message, "History clear!")
        return
    hist, rply = respond(hist, question)
    bot.reply_to(message, rply)


print('ready')
bot.infinity_polling()
