!pip install beautifulsoup4 requests transformers
from googlesearch import search
from transformers import pipeline, BertForQuestionAnswering, BertTokenizer
import torch
import requests
from bs4 import BeautifulSoup

# Function to get main text using BeautifulSoup
def extract_main_text(url):
    try:
        # Fetch HTML content of the web page
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract main text from the parsed HTML (considering div, span, h1, h2, h3, h4 tags)
        main_text = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4'])])

        return main_text

    except Exception as e:
        print(f"Error extracting main text: {e}")
        return None

# User input
user_prompt = input("Ask me something: ")

# Question-Answering using google-bert/bert-large-uncased-whole-word-masking-finetuned-squad
model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Web search using googlesearch library
search_results = list(search(user_prompt, num=5, stop=5))  # Get top 5 results
if not search_results:
    print("No search results found.")
    exit()

for result_url in search_results:
    print(f"\nChecking result from {result_url}")

    # Get content from the web page using BeautifulSoup
    context = extract_main_text(result_url)

    if context is None:
        print("Error getting content from the web page.")
        continue

    # Truncate context to fit within the model's maximum sequence length
    max_seq_length = 512 - len(tokenizer.encode(user_prompt, add_special_tokens=True))
    context = context[:max_seq_length]

    inputs = tokenizer.encode_plus(user_prompt, context, return_tensors="pt", truncation=True)
    start_positions, end_positions = model(**inputs).values()

    start_index = torch.argmax(start_positions)
    end_index = torch.argmax(end_positions) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index])
    print(f"Answer: {answer}")
