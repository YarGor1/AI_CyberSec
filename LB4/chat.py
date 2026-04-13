from ollama import chat
import ollama
print(ollama.list())

messages = []

gen_model = 'qwen3:0.6b'
#gen_model = 'smollm2:360m'

while True:
    mess = input('\nUser: ')
    
    messages.append({'role': 'user','content': mess})

    response = chat(gen_model, messages=messages, stream=True)
    print('Jarvis: ', end='')

    full_response_content = ""
    for chunk in response:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        full_response_content += content

    messages.append({'role': 'assistant', 'content': full_response_content})
