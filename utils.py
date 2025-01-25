# utils.py
import requests
import random
import shutil
import time
import os
import uuid
from gradio_client import Client
from aiogram.utils import exceptions
from aiogram.types import InputMediaPhoto
from config import UPLOADS_DIR, RESULTS_DIR, PROMTS_DIR
import tensorflow_hub as hub
import tensorflow as tf
import subprocess
import zipfile
from tqdm import tqdm
import tempfile
import base64
from states import StyleChoice, DialogStates  # Импортируем классы состояний
from aiogram import Bot, Dispatcher, executor, types

def download_file(url, path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    with open(path, 'wb') as file, tqdm(
            desc=path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def ensure_data_available(dataset_url, dataset_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    if not os.path.exists(os.path.join(extract_path, 'monet2photo')):
        if not os.path.exists(dataset_path):
            print("Загрузка датасета Monet2Photo...")
            download_file(dataset_url, dataset_path)
            print(f"Dataset downloaded to: {dataset_path}")
        
        print("Распаковка датасета...")
        extract_zip(dataset_path, extract_path)
        print(f"Dataset extracted to: {extract_path}")
        
def check_service_availability(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Сервис по адресу {url} доступен.")
        else:
            print(f"Сервис по адресу {url} вернул статус {response.status_code}.")
    except Exception as e:
        print(f"Ошибка при проверке доступности сервиса: {e}")

async def process_style_choice(user_id, state, photo_filename, bot, chosen_style, chosen_style_description, chosen_style_neodescription):
    # Получаем данные из состояния
    data = await state.get_data()
    chosen_style = data.get("chosen_style")
    chosen_style_description = data.get("chosen_style_description")
    chosen_style_neodescription = data.get("chosen_style_neodescription")
    from app2023 import photo_filename
    from aiogram import Bot, Dispatcher, executor, types


    # получаем рандомные числа
    seed_rnd = random.randint(1, 9611608263085119394)
    # Отправляем запрос к API
    client = Client("https://bcb7ff99d62f1ca8c8.gradio.live", serialize=False)
    #client = Client("http://127.0.0.1:7860/", serialize=False)
    try:
        import base64

        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        # Преобразуем изображения в base64
        image_base64 = image_to_base64(photo_filename)
        mask_base64 = image_to_base64(photo_filename)

        # Отправляем запрос к API
        result = client.predict(
            chosen_style_description, # str in 'parameter_10' Textbox component
            chosen_style_neodescription, # str in 'Negative Prompt' Textbox component
            ["Fooocus V2", "Fooocus Photograph", "Default (Slightly Cinematic)", "SAI Fantasy Art"], # List[str] in 'Selected Styles' Checkboxgroup component
            "Quality", # str in 'Performance' Radio component
            "1152×896", # <span style=\"color: color;\"> ∣ 1:2</span>", # str in 'Aspect Ratios' Radio component
            1, # int | float (numeric value between 1 and 32) in 'Image Number' Slider component
            seed_rnd, # str in 'Seed' Textbox component
            10, # int | float (numeric value between 0.0 and 30.0) in 'Image Sharpness' Slider component
            7, # int | float (numeric value between 1.0 and 30.0) in 'Guidance Scale' Slider component 
            "juggernautXL_version6Rundiffusion.safetensors", #juggernautXL_version6Rundiffusion.safetensors", # str (Option from: ['bluePencilXL_v050.safetensors', 'DreamShaper_8_pruned.safetensors', 'juggernautXL_version6Rundiffusion.safetensors', 'realisticStockPhoto_v10.safetensors']) in 'Base Model (SDXL only)' Dropdown component
            "realisticStockPhoto_v10.safetensors", #realisticStockPhoto_v10.safetensors", # str (Option from: ['None', 'bluePencilXL_v050.safetensors', 'DreamShaper_8_pruned.safetensors', 'juggernautXL_version6Rundiffusion.safetensors', 'realisticStockPhoto_v10.safetensors'])
            "sd_xl_offset_example-lora_1.0.safetensors", #sd_xl_offset_example-lora_1.0.safetensors", # in 'LoRA 1' str (Option from: ['None', 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sd_xl_offset_example-lora_1.0.safetensors']) in 'LoRA 1' Dropdown component
            1.2, # int | float (numeric value between -2 and 2) in 'Weight' Slider component
            "None", # in 'LoRA 2' str (Option from: ['None', 'SDXL_FILM_PHOTOGRAPHY_STYLE_BetaV0.4.safetensors', 'sd_xl_offset_example-lora_1.0.safetensors']) in 'LoRA 2' Dropdown component
            -2, # int | float (numeric value between -2 and 2) in 'Weight' Slider component
            "None", # in 'LoRA 3'
            -2, # int | float (numeric value between -2 and 2)
            "None", # in 'LoRA 4'
            -2, # int | float (numeric value between -2 and 2)
            "None", # in 'LoRA 5'
            -2, # int | float (numeric value between -2 and 2)
            True, # bool in 'Input Image' Checkbox component
            chosen_style_description, # str in 'parameter_83' Textbox component
            "Upscale (1.5x)", # str in 'Upscale or Variation:' Radio component
            f"data:image/png;base64,{mask_base64}", # str (filepath or URL to image) in 'Drag above image to here' Image component
            [], # List[str] in 'Outpaint Direction' Checkboxgroup component  Left,  ["Left", "Right", "Top", "Bottom"]
            f"data:image/png;base64,{mask_base64}",
            f"data:image/png;base64,{mask_base64}",
            0.5, # int | float (numeric value between 0.0 and 1.0) in 'Stop At' Slider component
            1, # int | float (numeric value between 0.0 and 2.0) in 'Weight' Slider component
            "PyraCanny", # str in 'Type' Radio component
            f"data:image/png;base64,{mask_base64}", # str (filepath or URL to image) in 'Image' Image component
            0.7, 	# int | float (numeric value between 0.0 and 1.0) in 'Stop At' Slider component
            1.139, # int | float (numeric value between 0.0 and 2.0) in 'Weight' Slider component
            "PyraCanny", # str in 'Type' Radio component
            f"data:image/png;base64,{mask_base64}", # str (filepath or URL to image) in 'Image' Image component
            0.7, # int | float (numeric value between 0.0 and 1.0) in 'Stop At' Slider component
            1.139, # int | float (numeric value between 0.0 and 2.0) in 'Weight' Slider component
            "PyraCanny", # str in 'Type' Radio component
            f"data:image/png;base64,{mask_base64}", # str (filepath or URL to image) in 'Image' Image component
            0.7, # int | float (numeric value between 0.0 and 1.0) in 'Stop At' Slider component
            1.139, # int | float (numeric value between 0.0 and 2.0) in 'Weight' Slider component
            "PyraCanny", # str in 'Type' Radio component
            fn_index=23
        )
        await process_result(user_id, state, result, bot, chosen_style, chosen_style_description, chosen_style_neodescription)
        
        #result = await client.predict(*service_params)
        #result = await client.view_api()
        # Разбираем кортеж на отдельные переменные
        #result_str_value_4 = str(result[0])
        #result_str_preview = str(result[1])
        #result_str_gallery = str(result[2])
        
        # # Выводим результаты
        #print("Ответ API (value_4):", result_str_value_4)
        #print("Ответ API (Preview):", result_str_preview)
        #print("Ответ API (Gallery):", result_str_gallery)
    except Exception as e:
        print("Ошибка при запросе API:", e)

async def process_result(user_id: int, state, result: tuple, bot, chosen_style, chosen_style_description, chosen_style_neodescription):
    max_retries = 15
    retry_delay = 10  # seconds
    #print("Результат от API:", result)
    print("Длина кортежа result:", len(result))
    try:
        # Проверяем, видимо ли изображение в Gallery
        if result[2]['visible']:
            image_info = result[2]['value'][0]
            #print (image_info)
            # Получаем путь к файлу изображения
            image_path = image_info.get('name')
            if image_path and os.path.isfile(image_path):
                # Пример: Создаем путь для сохранения обработанного изображения
                result_folder = os.path.join(RESULTS_DIR, str(user_id))
                os.makedirs(result_folder, exist_ok=True)
                result_filename2 = os.path.join(result_folder, f"{str(uuid.uuid4())}.png")
                #result_filename2=image_path
                # Переносим файл изображения в папку с результатами для экономии места
                #shutil.move(image_path, result_filename2)
                print (image_path)
                print (result_filename2)
                shutil.move(image_path, result_filename2)
#                # Ждем пока файл полностью переместится
                time.sleep(1)  # Задержка на 1 секунду

        # Отправляем фото с подписью через бота
        with open(result_filename2, 'rb') as photo:
            retries = 0
            while retries < max_retries:
                try:
                    with open(result_filename2, 'rb') as photo:
                        await bot.send_photo(
                            chat_id=user_id,
                            photo=photo,
                            caption=f"*Выбран стиль: *{chosen_style}\n"
                                    f"*Промт: *{chosen_style_description}\n"
                                    f"*Негативный промт: *{chosen_style_neodescription}\n",
                            parse_mode="Markdown"
                        )
                    print("Изображение успешно отправлено пользователю id:" + str(user_id))
                    break  # Выходим из цикла, если отправка прошла успешно
                except exceptions.RetryAfter as e:
                    print(f"Ошибка: Бот требует повторную отправку через {e.timeout} секунд.")
                    retries += 1
                    time.sleep(e.timeout)  # Ждем указанное количество секунд перед повторной попыткой
                except exceptions.TelegramAPIError as e:
                    if "ClientOSError" in str(e) and "[WinError 64]" in str(e):
                        print("Ошибка: Сетевое имя более недоступно. Повторная попытка.")
                    elif "read of closed file" in str(e):
                        print("Ошибка: Файл уже закрыт. Повторная попытка.")
                    else:
                        print(f"Ошибка при отправке изображения: {e}")
                        break  # Прекращаем повторные попытки, если возникла другая ошибка

    except Exception as e:
        print(f"Ошибка при обработке результата пользователя {user_id}:", e)

    # Возвращаемся в начальное состояние
    await DialogStates.Continue.set()

async def apply_style(user_id, state, processing_message_id, photo_filename, model, bot):
    # Загружаем исходное изображение
    original_image_data = tf.io.read_file(photo_filename)
    original_image_data = tf.image.decode_image(original_image_data)
    # Преобразуем исходное изображение в тензор [batch_size, height, width, 3]
    original_image_data = tf.image.convert_image_dtype(original_image_data, tf.float32)
    original_image_data = tf.image.resize(original_image_data, (512, 512))
    original_image_data = tf.expand_dims(original_image_data, 0)
    # print(original_image_data.shape)

    # Создаем список тензоров стилей
    style_image_paths = [os.path.join(PROMTS_DIR, f"{i}.jpg") for i in range(1, 10)]
    style_images = [tf.image.decode_image(tf.io.read_file(style_image_path)) for style_image_path in style_image_paths]
    style_images = [tf.image.convert_image_dtype(style_image, tf.float32) for style_image in style_images]
    style_images = [tf.image.resize(style_image, (512, 512)) for style_image in style_images]
    # # лог размерности
    # for style_image in style_images:
    #     print(style_image.shape)
    # Применяем каждый стиль к изображению
    stylized_images = []
    for i, style_image in enumerate(style_images, start=1):
        # outputs = model(original_image_data, tf.expand_dims(style_image, 0))
        # stylized_image = outputs[0]
        stylized_image = model(original_image_data, tf.expand_dims(style_image, 0))[0]
        # Конвертируем тензор обратно в изображение
        stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)
        #print(stylized_image.shape) # проверка (1, 256, 256, 3)
        #print(stylized_image.dtype) # проверка <dtype: 'uint8'>
        stylized_image = tf.squeeze(stylized_image)
        stylized_image_data = tf.io.encode_png(stylized_image).numpy()
        # Сохраняем стилизованное изображение на диск
        # Пример: Создаем путь для сохранения обработанного изображения
        result_folder = os.path.join(RESULTS_DIR, str(user_id))
        os.makedirs(result_folder, exist_ok=True)
        result_filename = os.path.join(result_folder, f"{str(uuid.uuid4())}_style_{i}.png")
        with open(result_filename, 'wb') as f:
            f.write(stylized_image_data)
        # Добавляем путь к сохраненному изображению в массив
        #print (result_filename)
        stylized_images.append(result_filename)
    # Проверяем, есть ли стилизованные изображения
    if stylized_images:
        # Редактируем предыдущее сообщение с текстом
        await bot.edit_message_text(
            chat_id=user_id,
            message_id=processing_message_id,
            text=f"*Выбран стиль из 9 промт фото *",
            parse_mode="Markdown"
        )
        media = [InputMediaPhoto(media=open(stylized_image, 'rb')) for stylized_image in stylized_images]
        await bot.send_media_group(chat_id=user_id, media=media)
        # Переходим в следующее состояние
        print("Отправлены стилизованные фото id: " + str(user_id))
    else:
        # Обработка случая, когда нет стилизованных изображений (если нужно что-то выполнить в этом случае)
        pass
        #DialogStates.Continue.set()

async def process_style_transfer(content_image_path, style_image_path):
    # Запуск Streamlit скрипта для обработки изображений
    result = subprocess.run(["streamlit", "run", "app.py", "--", content_image_path, style_image_path], capture_output=True, text=True)

    # Получение пути к сохраненному результату
    result_filename = result.stdout.strip().split()[-1]
    return result_filename
    