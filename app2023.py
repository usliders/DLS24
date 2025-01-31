import os
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import base64
import uuid
from io import BytesIO
from config import UPLOADS_DIR, RESULTS_DIR, PROMTS_DIR, API_TOKEN
from states import StyleChoice, DialogStates  # Импортируем классы состояний
from aiogram.types import InputMediaPhoto
from utils import check_service_availability, process_style_choice, process_result, apply_style
import random
from gradio_client import Client
import tensorflow as tf

# Отключаем лишние сообщения TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Определение размера изображения
desired_size = (512, 512)

# Обработка нажатий на кнопки выбора стиля
async def handle_style_choice_callback(query: types.CallbackQuery, state: FSMContext, bot: Bot, model):
    global chosen_style
    global chosen_style_description
    global chosen_style_neodescription
    user_id = query.from_user.id
    chosen_style = query.data

    # Проверяем, что пользователь выбрал стиль
    if not chosen_style:
        await query.answer("Пожалуйста, выберите стиль")
        return

    # Обрабатываем выбор стиля и сохраняем его в состоянии
    await state.update_data(chosen_style=chosen_style)

    if chosen_style in ["zhara", "sci", "aqvarel", "abstraction", "alien", "lego", "cat", "anticat"]:
        if chosen_style == "zhara":
            chosen_style_description = (
                "A beautiful fiery plumage of neon-bright colors sits on a fir branch ,"
                "the background is a winter fir forest, a full textured moon in the sky ,"
                "a firebird illuminates everything around itself, a long fractal fiery tail "
                "and crest emanate from the bird, the colors are golden orange black yellow blue red, magic "
                "brightness, hyperrealism, hyperdetalization, high quality, photorealism "
                "cinematography, professional photography, clear focus, 5D, three-dimensional "
                "drawing, UHD, ray tracing"
            )
            chosen_style_neodescription = ("low quality, bad hands, bad eyes, cropped, missing fingers, extra digit")
        elif chosen_style == "sci":
            chosen_style_description = (
                "sci-fi style {prompt} . futuristic, technological, alien worlds, space themes, "
                "advanced civilizations"
            )
            chosen_style_neodescription = (
                "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, historical, medieval, "
                "low quality, bad hands, bad eyes, cropped, missing fingers, extra digit"
            )
        elif chosen_style == "aqvarel":
            chosen_style_description = (
                "watercolor painting {prompt} . vibrant, beautiful, painterly, detailed, textural, artistic"
            )
            chosen_style_neodescription = (
                "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy, "
                "low quality, bad hands, bad eyes, cropped, missing fingers, extra digit"
            )
        elif chosen_style == "abstraction":
            chosen_style_description = (
                "abstract style {prompt} . non-representational, colors and shapes, "
                "expression of feelings, imaginative, highly detailed"
            )
            chosen_style_neodescription = (
                "realistic, photographic, figurative, concrete, "
                "low quality, bad hands, bad eyes, cropped, missing fingers, extra digit"
            )
        elif chosen_style == "alien":
            chosen_style_description = (
                "Xenomorph with translucent skin. Caustics. Prismatic light, detailed and "
                "intricate environment, artstation, concept art"
            )
            chosen_style_neodescription = ("low quality, bad hands, bad eyes, cropped, missing fingers, extra digit")
        elif chosen_style == "lego":
            chosen_style_description = (
                "A lego box with an image of (персонаж)"
            )
            chosen_style_neodescription = ("low quality, bad hands, bad eyes, cropped, missing fingers, extra digit")
        elif chosen_style == "cat":
            chosen_style_description = (
                "Intricate fluffy kitten Stevie hugs a large detailed magical glowing "
                "glass ball in a pile of colorful wires of a purple room with gray floor "
                "furniture; large amber eyes; Christmas tree; New Year's mood + coherent; "
                "high-octane render; a bunch of colorful wires; intricate concept art; "
                "mysticism; cyberpunk; cinematic; pastel colors; dark botanical cyberpunk; "
                "lumen lighting; lumen radiance"
            )
            chosen_style_neodescription = ("low quality, bad hands, bad eyes, cropped, missing fingers, extra digit")
        elif chosen_style == "anticat":
            chosen_style_description = (
                "war dog, in the style of vray, klaus wittmann, nikolai lockertsen, detailed "
                "character expressions, francesco solimena, studio portrait, hyper-realistic"
            )
            chosen_style_neodescription = ("low quality, bad hands, bad eyes, cropped, missing fingers, extra digit")

        # Обновляем состояние, передавая выбранные стиль, описание
        await state.update_data(
            chosen_style=chosen_style,
            chosen_style_description=chosen_style_description,
            chosen_style_neodescription=chosen_style_neodescription
        )
        await bot.edit_message_text(
            chat_id=query.from_user.id,
            message_id=query.message.message_id,
            text=f"*Выбран стиль: *{chosen_style}\n"
            f"*Промт: *{chosen_style_description}\n"
            f"*Негативный промт: *{chosen_style_neodescription}\n\n*Ожидайте обработку* (обычно 2-7 минут)",
            parse_mode="Markdown",
            reply_markup=None  # None, чтобы удалить кнопки
        )

        # Переходим в следующее состояние (или выполняем другие действия)
        await StyleChoice.ChoosingNextStep.set()
        await process_style_choice(user_id, state, photo_filename, bot, chosen_style, chosen_style_description, chosen_style_neodescription)

        # Удаляем предыдущее сообщение
        await bot.delete_message(chat_id=user_id, message_id=query.message.message_id)

    else:
        if chosen_style == "stile9":
            chosen_style_description = ("9 разных стилей")
            chosen_style_neodescription = ("low quality")

            # Удаляем предыдущее сообщение
            await bot.delete_message(chat_id=user_id, message_id=query.message.message_id)

            # Отправляем новое сообщение с изображением и текстовой подписью
            # Определение описаний для каждого стиля
            chosen_style_description = {
                1: "Venus_in_the_forge_of_Vulcan",
                2: "Alfons_Mucha",
                3: "СССР Фантасмогория",
                4: "Belgian_linen",
                5: "Circles_in_a_Circle",
                6: "British_Museum",
                7: "Moderne_de_la_Ville_de_Paris",
                8: "Portrait_de",
                9: "Vertical_Still_Life",
            }
            media = []
            style_image_base = os.path.join(PROMTS_DIR, "{i}.jpg")
            for i in range(1, 10):  # 9 изображений с названиями 1.jpg, 2.jpg, ..., 9.jpg
                style_image_path = style_image_base.format(i=i)
                caption = f"Стиль {i}: {chosen_style_description.get(i, 'Unknown')}"

                # Открываем изображение с использованием Pillow
                image = Image.open(style_image_path)
                # Изменяем размер изображения
                resized_image = image.resize(desired_size)
                # Перезаписываем изображение в том же файле
                resized_image.save(style_image_path)
                media.append(InputMediaPhoto(media=open(style_image_path, 'rb'), caption=caption))

            # Отправляем сообщение с 9 изображениями
            await bot.send_media_group(chat_id=user_id, media=media)
            processing_message = await bot.send_message(
                chat_id=query.from_user.id,
                text=f"*Выбран стиль из 9 промт фото *\n"
                f"Ожидайте обработку",
                parse_mode="Markdown"
            )
            processing_message_id = processing_message.message_id
            # Переходим в следующее состояние
            print("Отправлены промт фото")
            await StyleChoice.ChoosingNextStep.set()
            # Обрабатываем 9 стилей
            # Получаем массив путей к стилизованным изображениям
            # stylized_images = await apply_style(user_id, state)
            await apply_style(user_id, state, processing_message_id, photo_filename, model, bot)
            
async def handle_style_transfer(message: types.Message, state: FSMContext, bot: Bot):
    user_id = message.from_user.id
    unique_folder = os.path.join(UPLOADS_DIR, str(user_id))
    os.makedirs(unique_folder, exist_ok=True)

    # Загрузка контентного изображения
    content_image_file = await bot.get_file(message.photo[-1].file_id)
    content_image_path = os.path.join(unique_folder, f"content_{str(uuid.uuid4())}.jpg")
    await content_image_file.download(destination=content_image_path)

    # Загрузка стилевого изображения
    style_image_file = await bot.get_file(message.photo[-2].file_id)
    style_image_path = os.path.join(unique_folder, f"style_{str(uuid.uuid4())}.jpg")
    await style_image_file.download(destination=style_image_path)

    # Обработка изображений с помощью Streamlit
    result_filename = await process_style_transfer(content_image_path, style_image_path)

    # Отправка результата пользователю
    with open(result_filename, 'rb') as photo:
        await bot.send_photo(chat_id=user_id, photo=photo, caption="Вот ваше стилизованное изображение!")

    # Удаление временных файлов
    os.remove(content_image_path)
    os.remove(style_image_path)
    os.remove(result_filename)

async def handle_one_photo(message: types.Message, state: FSMContext, bot: Bot):
    global encoded_image
    global photo_filename
    user_id = message.from_user.id
    unique_folder = os.path.join(UPLOADS_DIR, str(user_id))
    os.makedirs(unique_folder, exist_ok=True)
    print("Отправка photos..." + str(user_id))

    # Проверяем наличие кнопок в предыдущем сообщении
    if message.reply_markup and isinstance(message.reply_markup, types.ReplyKeyboardMarkup):
        # Удаляем предыдущее сообщение с кнопками
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id - 1)

    # Сохраняем изображение
    file_info = await bot.get_file(message.photo[-1].file_id)
    file_path = file_info.file_path
    photo_url = f"https://api.telegram.org/file/bot{API_TOKEN}/{file_path}"
    photo_filename = os.path.join(unique_folder, rf"{str(uuid.uuid4())}.jpg")

    print(f"File path: {file_path}")
    print(f"Photo URL: {photo_url}")
    print(f"Photo filename: {photo_filename}")

    # Используем BytesIO для записи файла
    file_data = await bot.download_file(file_path)
    file_bytes = BytesIO(file_data.read())

    with open(photo_filename, 'wb') as photo_file:
        photo_file.write(file_bytes.getvalue())

    # Проверка, что файл успешно сохранен
    if os.path.exists(photo_filename):
        print(f"Фото успешно сохранено: {photo_filename}")
    else:
        print(f"Ошибка при сохранении фото: {photo_filename}")

    # Прочитайте изображение из файла и закодируйте его в base64
    with open(photo_filename, "rb") as image_file:
        original_image_data = image_file.read()
        encoded_image = base64.b64encode(original_image_data).decode("utf-8")

    print(f"Encoded image: {encoded_image[:10]}...")  # Выводим первые 100 символов для отладки

    # Переходим в состояние выбора стиля
    await StyleChoice.ChoosingStyle.set()

    # Создаем кнопки с названиями стилей
    style_buttons = [
        types.InlineKeyboardButton(text="Жар-птица", callback_data="zhara"),
        types.InlineKeyboardButton(text="Фантастика", callback_data="sci"),
        types.InlineKeyboardButton(text="Акварель", callback_data="aqvarel"),
        types.InlineKeyboardButton(text="Абстракция", callback_data="abstraction"),
        types.InlineKeyboardButton(text="Чужой", callback_data="alien"),
        types.InlineKeyboardButton(text="Lego Marvel", callback_data="lego"),
        types.InlineKeyboardButton(text="Котёнок", callback_data="cat"),
        types.InlineKeyboardButton(text="Пёс", callback_data="anticat"),
        types.InlineKeyboardButton(text="9 стилей", callback_data="stile9"),
    ]

    # Добавляем кнопки в сообщение
    keyboard_markup = types.InlineKeyboardMarkup().add(*style_buttons)
    await message.answer("Выберите стиль:", reply_markup=keyboard_markup)

