import os
import base64
from nicegui import ui
from huggingface_hub import InferenceClient

# Читаем токен из переменных окружения Replit
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not HF_TOKEN:
    print("⚠️ ВНИМАНИЕ: Токен HF_TOKEN не найден в Secrets!")

client = InferenceClient(token=HF_TOKEN)
TEXT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
IMAGE_MODEL = "stabilityai/stable-diffusion-2-1"

def classify_intent(prompt: str) -> str:
    prompt_lower = prompt.lower()
    image_keywords = ["нарисуй", "изображение", "картинку", "фото", "создай изображение", "сгенерируй", "draw", "image", "picture", "иллюстрация", "скетч"]
    if any(keyword in prompt_lower for keyword in image_keywords):
        return "image"
    return "text"

async def handle_send():
    prompt = input_field.value.strip()
    if not prompt:
        return

    with chat_container:
        ui.label(prompt).style('align-self: flex-end; background: #e3f2fd; padding: 12px; border-radius: 12px; margin: 5px; max-width: 80%; font-size: 16px;')

    input_field.value = ""
    input_field.disable()

    with chat_container:
        loading_label = ui.label("Анализирую запрос...").style('align-self: flex-start; color: gray; margin: 5px; font-style: italic;')
        spinner = ui.spinner(size='md', color='primary').style('align-self: flex-start; margin: 5px;')

    intent = classify_intent(prompt)
    loading_label.text = f"Распознан тип задачи: {'Генерация изображения 🎨' if intent == 'image' else 'Генерация текста 📝'}. Обрабатываю..."

    try:
        if intent == "image":
            image_bytes = client.text_to_image(prompt, model=IMAGE_MODEL)
            b64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{b64_image}"
            loading_label.delete()
            spinner.delete()
            with chat_container:
                ui.image(image_url).style('max-width: 400px; border-radius: 12px; align-self: flex-start; margin: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);')
        else:
            text_response = client.text_generation(prompt, model=TEXT_MODEL, max_new_tokens=250)
            loading_label.delete()
            spinner.delete()
            with chat_container:
                ui.label(text_response).style('align-self: flex-start; background: #f1f1f1; padding: 12px; border-radius: 12px; margin: 5px; max-width: 80%; white-space: pre-wrap; font-size: 16px;')
    except Exception as e:
        loading_label.delete()
        spinner.delete()
        with chat_container:
            ui.label(f"⚠️ Ошибка: {str(e)}. Проверьте HF_TOKEN в Secrets.").style('align-self: flex-start; color: #d32f2f; margin: 5px; background: #ffebee; padding: 10px; border-radius: 8px;')
    finally:
        input_field.enable()
        input_field.focus()
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight);')

ui.add_css('''
    body { background-color: #f5f7fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .chat-container { height: 65vh; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; scroll-behavior: smooth; }
''')

with ui.column().classes('w-full max-w-3xl mx-auto q-pa-md'):
    ui.label('🤖 Мультимодальный Агент').classes('text-h5 text-center q-mb-md text-primary')
    ui.label('Введите запрос, я сам пойму: нужен текст или картинка!').classes('text-subtitle1 text-center q-mb-md text-grey-7')
    chat_container = ui.column().classes('chat-container rounded-borders shadow bg-white')
    with ui.row().classes('w-full q-mt-md items-center'):
        input_field = ui.input(placeholder='Например: "Нарисуй киберпанк город" или "Расскажи о квантовой физике"').classes('flex-grow').props('outlined dense rounded').on('keydown.enter', handle_send)
        ui.button('Отправить', on_click=handle_send).props('unelevated color=primary rounded').classes('q-ml-sm')

port = int(os.getenv('PORT', 8080))
ui.run(host='0.0.0.0', port=port, title="Multimodal Agent", reload=False)
