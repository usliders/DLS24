# main.py
import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ParseMode
from config import API_TOKEN, UPLOADS_DIR, RESULTS_DIR, PROMTS_DIR, LOG_FILE_PATH
from app2023 import handle_style_choice_callback, handle_style_transfer, handle_one_photo
from app2024 import handle_content_photo, handle_style_photo
from utils import check_service_availability
import tensorflow_hub as hub
import tensorflow as tf
from functools import partial
from states import StyleChoice, DialogStates  # Импортируем классы состояний
import os
import uuid
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Fooocus 2.1.701
# pip install tensorflow==2.10.0 # windows cuda 12.6 cuddn 9.3 drivers 560.94
# датасеты https://github.com/junyanz/CycleGAN/blob/master/datasets/download_dataset.sh
# pip install realesrgan # ля увелечения картинки на выходе
# Установите уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает все сообщения TensorFlow, кроме ошибок
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Отключает логирование TensorFlow
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Игнорировать предупреждения пользователя

# Инициализация бота и диспетчера
bot = Bot(
    token=API_TOKEN,
    timeout=420
)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Загружаем модель
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Глобальные переменные
photo_filename = None
file_data = None
file_bytes = None
file_path = None
photo_file = None
encoded_image = None
original_image_data = None
photo_url = None
chosen_style_description = None
chosen_style_neodescription = None
image_data_uri = None
seed_rnd = None
result_image_file = None
result_filename = None
result_filename2 = None
processing_message_id = None
desired_size = (512, 512)

# Создаем подпапки, если они не существуют
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

async def send_welcome(message: types.Message, state: FSMContext):
    await DialogStates.Start.set()
    await message.answer("Привет! Этот бот умеет стилизовать изображения.")
    await DialogStates.Continue.set()

    # Добавляем две кнопки для выбора варианта
    reply_markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    variant1_button = types.KeyboardButton("две фото (проект 24)")
    variant2_button = types.KeyboardButton("одно фото (проект 23)")
    reply_markup.add(variant1_button, variant2_button)

    await message.answer("Выберите вариант:", reply_markup=reply_markup)
    await DialogStates.ChoosingVariant.set()
    
# Обработчик кнопки "Начать"
async def handle_start_button(message: types.Message, state: FSMContext):
    #await state.finish()
    await send_welcome(message, state)

# Обработчик выбора варианта
async def handle_variant_choice(message: types.Message, state: FSMContext):
    logging.info(f"User {message.from_user.id} chose variant: {message.text}")
    user_id = message.from_user.id
    if message.text == "две фото (проект 24)":
        await state.update_data(variant="two_photos")
        await message.answer("Выбран \"Перенос стиля\"", reply_markup=types.ReplyKeyboardRemove())
        await message.answer("Отправьте Контентное/исходное изображение.")
        await DialogStates.WaitingForContentPhoto.set()
    elif message.text == "одно фото (проект 23)":
        await state.update_data(variant="one_photo")
        await message.answer("Отправьте одну фотографию для улучшения качества.", reply_markup=types.ReplyKeyboardRemove())
        await DialogStates.WaitingForPhoto.set()
    else:
        await message.answer("Что то пошло не так. Пожалуйста, начните через МЕНЮ - СТАРТ.")

# Обработчик для управления вариантами обработки фотографий
async def handle_photos(message: types.Message, state: FSMContext, bot: Bot):
    user_data = await state.get_data()
    variant = user_data.get('variant')

    if variant == "two_photos":
        current_state = await state.get_state()
        if current_state == "DialogStates:WaitingForContentPhoto":
            await DialogStates.WaitingForStylePhoto.set()
            await handle_content_photo(message, state, bot)
        elif current_state == "DialogStates:WaitingForStylePhoto":
            await message.answer("Спасибо! Обрабатываю фото...")
            # Здесь вы можете добавить логику обработки фото
            await DialogStates.Start.set()
            await handle_style_photo(message, state, bot)
    elif variant == "one_photo":
        await handle_one_photo(message, state, bot)
    else:
        await message.answer("Пожалуйста, выберите один из предложенных вариантов.")

# Обработчик для возврата в начальное состояние после отправки измененных фотографий
async def return_to_start(message: types.Message, state: FSMContext):
    await send_welcome(message, state)

# Регистрация обработчиков
def register_handlers(dp: Dispatcher):
    dp.register_message_handler(send_welcome, commands=['start', 'help'])
    dp.register_message_handler(handle_start_button, lambda message: message.text.lower() == 'начать сначало', state="DialogStates:finish")
    dp.register_message_handler(handle_variant_choice, lambda message: message.text in ["две фото (проект 24)", "одно фото (проект 23)"], state="*")
    dp.register_message_handler(partial(handle_photos, bot=bot), content_types=types.ContentTypes.PHOTO, state="*")
    dp.register_callback_query_handler(partial(handle_style_choice_callback, bot=bot, model=model), lambda query: True, state="StyleChoice:ChoosingStyle")
    dp.register_message_handler(handle_one_photo, content_types=types.ContentTypes.PHOTO, state="DialogStates:WaitingForPhoto")
    dp.register_message_handler(return_to_start, commands=['start', 'help'], state="*")
    dp.register_message_handler(handle_content_photo, content_types=types.ContentTypes.PHOTO, state="DialogStates:WaitingForContentPhoto")
    dp.register_message_handler(handle_style_photo, content_types=types.ContentTypes.PHOTO, state="DialogStates:WaitingForStylePhoto")
    #dp.register_message_handler(handle_style_photo, lambda message: message.text.lower() == 'начать сначало', state="DialogStates:finish")
# Определяем асинхронную функцию on_startup, которая будет вызываться при запуске бота
async def on_startup(dp):
    # Отправляем сообщение с текстом "Бот запущен!" на chat_id = 153330435
    await bot.send_message(chat_id=153330435, text="🤖  Бот запущен!")

# Проверяем, запущен ли скрипт как главная программа
if __name__ == '__main__':
    # Запускаем процесс опроса бота с помощью метода executor.start_polling()
    # Передаем объект dp и функцию on_startup в качестве параметров
    register_handlers(dp)
    executor.start_polling(dp, on_startup=on_startup)
