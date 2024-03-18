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


bot = telebot.TeleBot("6449634010:AAFMpPNmy1NEyxa45oVfjSsY_D1fDZxgQmo")

alia = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/checkpoint-79000/content/drive/MyDrive/gpt-oasst/checkpoint-79000").to('cuda')

alia_tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/checkpoint-79000/content/drive/MyDrive/gpt-oasst/checkpoint-79000", device="cuda:0")


ROLE_TOKEN = {
    'system': '[SYS]',
    'user':   '[USR]',
    'web':    '[WEB]',
    'bot':    '[BOT]'
}
S_END = '[/]'
S_PAD = '[_]'

# print(text)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad", device="cuda:0")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to('cuda')


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



def google_ans(url):

  search_results = googlesearch.search(url, num=1, stop=1)
  if search_results:
    website_url = next(search_results)
    url = website_url
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    # kill all script and style elements
    for script in soup(["script", "style"]):
      script.extract()    # rip it out
      # get text
      text = soup.get_text()
      # break into lines and remove leading and trailing space on each
      lines = (line.strip() for line in text.splitlines())
      # break multi-headlines into a line each
      chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
      # drop blank lines
      text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


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


def get_all_res (question):
  question = question
  text = f"""{google_ans(question)}"""
  max_segment_length = 512  # Maximum input length for BERT
  text_segments = split_text(text, max_segment_length)
  best_answer = ""  # Initialize variables to store the best answer and its scores
  best_start_score = -float('inf')
  best_end_score = -float('inf')
  anss = []

  for i, segment in enumerate(text_segments):
      inputs = tokenizer(question, segment, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to('cuda')
      with torch.no_grad():
          outputs = model(**inputs)

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
          best_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
          print(best_answer)
          anss.append(best_answer)
  best_answer = '; '.join(anss)
  print(best_answer)
  input_text = f"[USR]{question}[/][WEB]{best_answer}[/][BOT]"
  input_ids = alia_tokenizer(input_text, return_tensors="pt").to('cuda')


  alia_outputs = alia.generate(**input_ids, max_length = 512, repetition_penalty = 10., encoder_repetition_penalty = 10., eos_token_id = alia_tokenizer(S_END)['input_ids'][0])
  yt = alia_tokenizer.decode(alia_outputs[0])

  print(yt[(yt.index('[BOT]') + 5):-3])

  return yt[(yt.index('[BOT]') + 5):-3]

  # print(yt[(yt.index('[BOT]') + 5):-3])


# print(get_all_res("who is Elon Musk"))



@bot.message_handler(func=lambda message: True)
def echo_all(message):
    question = message.text
    rply = get_all_res(question)
    bot.reply_to(message, rply)


bot.polling()



# Print the best answer found
# print("Best Answer:", best_answer)
# print("Start Score:", best_start_score)
# print("End Score:", best_end_score)

