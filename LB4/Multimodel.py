from ollama import chat
from pathlib import Path

# 1. Вказуємо шлях
path = input('Будь ласка, введіть шлях до зображення: ')

# Перевіряємо, чи існує файл
if not Path(path).exists():
    print("Помилка: Файл не знайдено!")
    exit()

model_name = 'moondream:1.8b'
#model_name = 'llava:7b'

print('Jarvis аналізує: ', end='', flush=True)

# 3. Запускаємо чат зі стрімінгом
response = chat(
    model=model_name,
    messages=[
        {
            'role': 'user',
            'content': 'What is in this image? Be concise.',
            'images': [path],
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk['message']['content'], end='', flush=True)

print()
