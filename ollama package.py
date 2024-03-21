import pandas as pd
from datasets import load_dataset
import ollama

# Import Product Reviews Clusters

nike_airmax = pd.read_excel("Amazon Nike Men's Air Max 2017/Airmax Clustered Reviews.xlsx")
nike_airmax_clust1 = nike_airmax[nike_airmax['kmeans_sent_3clust'] == 0]
nike_airmax_clust2 = nike_airmax[nike_airmax['kmeans_sent_3clust'] == 1]
nike_airmax_clust3 = nike_airmax[nike_airmax['kmeans_sent_3clust'] == 2]

clust_1 = ' '.join(nike_airmax_clust1['review'])
clust_2 = ' '.join(nike_airmax_clust2['review'])
clust_3 = ' '.join(nike_airmax_clust3['review'])


# Text Summarization

def llm_summarize(llm, product, reviews):
    response = ollama.generate(model = llm, prompt = f'Please give me a summary of the following product reviews submitted by those who have purchased {product}: {reviews}')
    return response['response']

summarized_nike_clust1_llama2 = llm_summarize(llm = 'llama2', product = "Nike Men's Aimax 2017", reviews = clust_1)
print(summarized_nike_clust1_llama2['response'])

message = ollama.generate(model = 'llama2', prompt = f"Consider a number of customers who bought Nike Airmax 2017 and have the following opinions: {summarized_nike_clust1_llama2}. Now, the marketing team aims to increase customers' lifetime value so that Amazon can offer complementary products—for instance, a Nike short-sleeved T-shirt. Please generate 10 short messages that is going to be sent to the customer from marketing team of Amazon. Use the summarized opinions for customization and have some creativity to enhance the quality and persuasiveness of the message. Customers who would receive these messages must be motivated to start working out wearing Nike's products. You must address customer's problems and indicate that Amazon appreciates negative feedbacks as the cornerstone of its customer service. Please do not use more than 200 characters:")
print(message['response'])

feedback = ollama.chat(model = 'llama2', messages=[{
        'role': 'user', 
        'content': f"Suppose you are an Amazon customer who have purchased Nike's Men Airmax 2017 running shoes. Your opinions about the product are as follows: {summarized_nike_clust1_llama2}. Now, Amazon has sent you this text message via the app: {message}. Do you find such a message persuasive to search through Amazon again? What do you think of its tone?"
        }])

print(feedback['message']['content'])

summarized_nike_clust2_llama2 = llm_summarize(llm = 'llama2', product = "Nike Men's Aimax 2017", reviews = clust_2)
print(summarized_nike_clust2_llama2)

summarized_nike_clust3_llama2 = llm_summarize(llm = 'llama2', product = "Nike Men's Aimax 2017", reviews = clust_3)
print(summarized_nike_clust3_llama2)

summarized_nike_clust1_mistral = llm_summarize(llm = 'mistral', product = "Nike Men's Aimax 2017", reviews = clust_1)
print(summarized_nike_clust1_mistral)

summarized_nike_clust2_mistral = llm_summarize(llm = 'mistral', product = "Nike Men's Aimax 2017", reviews = clust_2)
print(summarized_nike_clust2_mistral)

summarized_nike_clust3_mistral = llm_summarize(llm = 'mistral', product = "Nike Men's Aimax 2017", reviews = clust_3)
print(summarized_nike_clust3_mistral)


# Text Generation

def llm_textgen(llm, product, reviews):
    response = ollama.chat(model = llm, messages=[{
        'role': 'user', 
        'content': f'Please generate a promotional text sending to customers with the following attitudes and experiences. Use inspiration content, which includes how {product} would benefit the customer after purchasing. As a marketing strategy, we aim to raise our retention rate: {reviews}'
        }])
    return response['message']['content']


generate_nike_clust1_llama2 = llm_textgen(llm = 'llama2', product = "Nike Men's Aimax 2017", reviews = summarized_nike_clust1_llama2)
print(generate_nike_clust1_llama2)

generate_nike_clust2_llama2 = llm_textgen(llm = 'llama2', product = "Nike Men's Aimax 2017", reviews = summarized_nike_clust2_llama2)
print(generate_nike_clust2_llama2)

generate_nike_clust3_llama2 = llm_textgen(llm = 'llama2', product = "Nike Men's Aimax 2017", reviews = summarized_nike_clust3_llama2)
print(generate_nike_clust3_llama2)

generate_nike_clust1_mistral = llm_textgen(llm = 'mistral', product = "Nike Men's Aimax 2017", reviews = summarized_nike_clust1_mistral)
print(generate_nike_clust1_mistral)

generate_nike_clust2_mistral = llm_textgen(llm = 'mistral', product = "Nike Men's Aimax 2017", reviews = summarized_nike_clust2_mistral)
print(generate_nike_clust2_mistral)

generate_nike_clust3_mistral = llm_textgen(llm = 'mistral', product = "Nike Men's Aimax 2017", reviews = summarized_nike_clust3_mistral)
print(generate_nike_clust3_mistral)


summaries_promotions = pd.DataFrame({
    'llm':['llama2', 'llama2', 'llama2', 'mistral', 'mistral', 'mistral'],
    'summary': [summarized_nike_clust1_llama2, summarized_nike_clust2_llama2, summarized_nike_clust3_llama2, summarized_nike_clust1_mistral, summarized_nike_clust2_mistral, summarized_nike_clust3_mistral],
    'generated_text': [generate_nike_clust1_llama2, generate_nike_clust2_llama2, generate_nike_clust3_llama2, generate_nike_clust1_mistral, generate_nike_clust2_mistral, generate_nike_clust3_mistral]
})

summaries_promotions.to_excel('summaries_promotions.xlsx')


# Read Previous Generated Messages
airmax_messages = pd.read_excel("Amazon Nike Men's Air Max 2017/ollama_summaries_promotions.xlsx")