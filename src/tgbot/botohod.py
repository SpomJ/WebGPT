
!pip install html2text pyTelegramBotAPI
import html2text
import telebot
import googlesearch
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from strip_markdown import strip_markdown
import requests



bot = telebot.TeleBot("6449634010:AAFMpPNmy1NEyxa45oVfjSsY_D1fDZxgQmo")


def answer_question(context, question):
    # Load pre-trained model and tokenizer
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)

    # Make prediction
    start_positions, end_positions = model(**inputs).values()

    # Get the answer text
    answer_start = torch.argmax(start_positions)
    answer_end = torch.argmax(end_positions) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer

def get_text(url):
    return html2text.HTML2Text().handle(requests.get(url).text)

# Example usage
def find_answer(question):
    # Use Google to search for a URL related to the question
    search_results = googlesearch.search(question, num=1, stop=1)

    if search_results:
        website_url = next(search_results)
        print("Found website URL:", website_url)

        text = get_text(website_url)

        return text




@bot.message_handler(func=lambda m: True)
def echo_all(message):
  question_text = message.text
  context_text = f'''{find_answer(question_text)}'''
  bot.reply_to(message, answer_question(question_text, context_text))

# question_text = input()
# context_text = f'''{find_answer(question_text)}'''

# result = answer_question(context_text, question_text)
# print(f"Answer: {result}")

# print(context_text)
bot.infinity_polling()

!pip install strip_markdown