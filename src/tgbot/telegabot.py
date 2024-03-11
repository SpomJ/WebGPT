# !pip install pyTelegramBotAPI
import telebot
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


import requests
import googlesearch
from bs4 import BeautifulSoup

def get_page(url):
    return requests.get(url).text

def extract_main_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    main_text = soup.get_text()
    return main_text

def find_answer(question):
    # Используем Google для поиска URL по вопросу
    search_results = googlesearch.search(question, num=1, stop=1)

    if search_results:
        website_url = next(search_results)
        print("Найден URL сайта:", website_url)

        # Получаем содержимое веб-страницы
        html_content = get_page(website_url)

        # Извлекаем весь текст с веб-страницы
        main_text = extract_main_text(html_content)

        return main_text
    else:
        return "Сайт не найден."

# # Пример использования:
# question = input("Введите вопрос: ")

# answer = find_answer(question)
# print("Ответ:", answer)


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

bot = telebot.TeleBot('6449634010:AAFMpPNmy1NEyxa45oVfjSsY_D1fDZxgQmo')


@bot.message_handler(commands=['start'])
def start(message):
    print(message)
    bot.send_message(message.chat.id,
                     'Hi, I am webGPT, your personal assistant. I possess knowledge as pretrained model and I also have an ability to send web requests about events in the world. Feel free to ask anything.')


@bot.message_handler()
def gen(message):
    input_text = "You are a chatbot that must, after analyzing the response from the Internet presented after the word assistant, answer the user's question presented after the word user. Your answer to the user's question should be based on the Internet response." + "\n user:" + message.text + "\n assistant" + find_answer(message.text)
    input_ids = tokenizer(input_text, return_tensors="pt")

    generation_config = GenerationConfig.from_pretrained("gpt2", num_beams=3, do_sample=True, max_new_tokens=128,
                                                         repetition_penalty=1.0)

    outputs = model.generate(**input_ids, generation_config=generation_config)
    bot.send_message(message.chat.id, text=tokenizer.decode(outputs[0]))


bot.infinity_polling()
