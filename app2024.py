# app2024.py
import os
import uuid
import logging
from aiogram import Bot, types
from aiogram.dispatcher import FSMContext
from config import UPLOADS_DIR, RESULTS_DIR, TIME_ASK
from nst import NeuralStyleTransfer
from aiogram.types import Message
import asyncio
from io import BytesIO
import tensorflow as tf
import gc

logger = logging.getLogger(__name__)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

async def handle_content_photo(message: types.Message, state: FSMContext, bot: Bot):
    try:
        user_id = message.from_user.id
        logger.info(f"Начало обработки контентного фото для пользователя {user_id}")
        
        # Создание папки пользователя
        user_dir = os.path.join(UPLOADS_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        logger.debug(f"Создана папка пользователя: {user_dir}")

        # Сохранение контентного фото
        file_info = await bot.get_file(message.photo[-1].file_id)
        content_filename = f"content_{uuid.uuid4()}.jpg"
        content_path = os.path.join(user_dir, content_filename)

        logger.info(f"Скачивание файла: {file_info.file_path} -> {content_path}")
        await bot.download_file(file_info.file_path, content_path)
        
        logger.info(f"Контентное изображение сохранено: {content_path}")
        await state.update_data(content_path=content_path)
        await message.answer("✅ Контентное изображение получено!")
        await message.answer("Теперь отправьте стилевое изображение.")
        await state.set_state("DialogStates:WaitingForStylePhoto")

    except Exception as e:
        logger.error(f"Ошибка обработки контентного фото: {str(e)}", exc_info=True)
        await message.answer("⚠ Произошла ошибка при обработке контентного изображения")

async def update_timer(message: Message, total_time: int):
    for remaining in range(total_time, 0, -5):
        try:
            await message.edit_text(f"🕒 Осталось: {remaining} сек...")
        except:
            pass
        await asyncio.sleep(5)

async def handle_style_photo(message: types.Message, state: FSMContext, bot: Bot):
    processor = None  # Объявляем переменную в начале
    try:
        user_id = message.from_user.id
        unique_folder = os.path.join(UPLOADS_DIR, str(user_id))
        os.makedirs(unique_folder, exist_ok=True)
        data = await state.get_data()
        content_path = data['content_path']
        
        # Сохранение стилевого изображения
        style_image_file = await bot.get_file(message.photo[-1].file_id)
        file_path = style_image_file.file_path        
        style_path = os.path.join(unique_folder, f"style_{uuid.uuid4()}.jpg")
        file_data = await bot.download_file(file_path)
        file_bytes = BytesIO(file_data.read())        
        with open(style_path, 'wb') as photo_file:
            photo_file.write(file_bytes.getvalue())
        
        logger.info(f"Стилевое изображение получено: {style_path}")

        # Запускаем таймер
        timer_msg = await message.answer("⏳ Начинаю обработку...")
        timer_task = asyncio.create_task(update_timer(timer_msg, TIME_ASK))

        # Основная обработка
        processor = NeuralStyleTransfer(img_size=512)
        result_image = await asyncio.get_event_loop().run_in_executor(
            None,
            processor.transfer_style,
            content_path,
            style_path
        )


        # Останавливаем таймер
        timer_task.cancel()
        await timer_msg.delete()

        # Сохранение и отправка результата
        result_folder = os.path.join(RESULTS_DIR, str(user_id))
        os.makedirs(result_folder, exist_ok=True)
        result_path = os.path.join(result_folder, f"{uuid.uuid4()}_style_2024.png")
        result_image.save(result_path)
        
        with open(result_path, 'rb') as photo:
            await message.answer_photo(photo, caption="🎉 Результат готов!")

    except Exception as e:
        logger.error(f"Ошибка обработки: {str(e)}", exc_info=True)
        await message.answer("⚠ Ошибка обработки. Попробуйте другие изображения.")
    finally:
        # Явное удаление процессора для вызова __del__
        if processor is not None:
            del processor
            gc.collect()
        await state.finish()