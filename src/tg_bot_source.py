# !pip install html2text pyTelegramBotAPI
from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, AutoModelForCausalLM
import torch
import googlesearch
import requests
import html2text
import telebot
from urllib.request import urlopen
from bs4 import BeautifulSoup
import telebot
from web import *
from model import *
from duckduckgo_search import DDGS

MODEL_PATH = ''

model = Model(MODEL_PATH)

# print(text)

bert = Model("bert-large-uncased-whole-word-masking-finetuned-squad")
bert.model = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", device="cuda:0")
bert.tokenizer = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to('cuda')

BOT_TOKEN = "6449634010:AAFMpPNmy1NEyxa45oVfjSsY_D1fDZxgQmo"

# Initialize Telebot
bot = telebot.TeleBot(BOT_TOKEN)


# def find_answer(question):
#     # Use Google to search for a URL related to the question
#     search_results = googlesearch.search(question, num=1, stop=1)

#     if search_results:
#         website_url = next(search_results)
#         print("Found website URL:", website_url)

#         text = get_text(website_url)


def split_text(text, max_segment_length):
    segments = []
    current_segment = ""
    for paragraph in text.split("\n"):  # Split by paragraphs for simplicity
        if len(current_segment) + len(paragraph) < max_segment_length:
            current_segment += paragraph + "\n"
        else:
            segments.append(current_segment.strip())
            current_segment = paragraph + "\n"
    if current_segment:  # Add the last segment
        segments.append(current_segment.strip())

    print(segments)
    return segments


# Load tokenizer and model


# Example text
# text = f"""google_ans{question}
# """


def results(question):
    text = search_full(question)
    max_segment_length = 512  # Maximum input length for BERT
    text_segments = split_text(text, max_segment_length//2)
    best_answer = ""  # Initialize variables to store the best answer and its scores
    best_start_score = -float('inf')
    best_end_score = -float('inf')
    anss = []

    for i, segment in enumerate(text_segments):
        inputs = tokenizer(
            question,
            segment,
            return_tensors="pt",
            truncation=True,
            max_length=512,
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
            best_answer = tokenizer.convert_tokens_to_string(
                tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end]))
            # print(best_answer)
            anss.append(best_answer)

    return = '; '.join(anss)

def respond(hist, q):
    hist = hist + [
        {'role': 'user', 'content': q},
        {'role': 'web', 'content': results(q)}]
    input_text = Model.fmt(hist)
    output = model.str_response(text)

    return output

    # print(yt[(yt.index('[BOT]') + 5):-3])


# print(get_all_res("who is Elon Musk"))


@bot.message_handler(func=lambda x: 1)
def echo_all(message):
    question = message.text
    rply = respond(question)
    bot.reply_to(message, rply)

print('ready')
bot.infinity_polling()

# Print the best answer found
# print("Best Answer:", best_answer)
# print("Start Score:", best_start_score)
# print("End Score:", best_end_score)
