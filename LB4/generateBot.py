from ollama import generate

mess = input('\nUser: ')

gen_model = 'qwen3:0.6b'
#gen_model = 'smollm2:360m'

response = generate(gen_model, mess)
print('Jarvis: ', end='')
print(response['response'])

