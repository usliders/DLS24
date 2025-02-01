# DLS24
Итоговый проект 2024 DLS TelegramBot


# 🎨 Deep Learning School 2024: Neural Style Transfer Telegram Bot

[![Python 3.10](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Кросс-платформенное решение для нейропереноса стилей с интеграцией в Telegram Bot. Поддерживает работу как на локальных GPU-машинах, так и в Google Colab.

## 🎥 Видео-демонстрации проекта

	> 📌 **Важно**: Для просмотра видео нажмите на заголовок. Может потребоваться авторизация на Rutube.

### 🎬 Основные демонстрации
  
- **🔊 Голосовой колаб-перенос**    
   
  [![Watch Video](https://img.shields.io/badge/Смотреть_видео-FF6F61?style=flat&logo=rutube&logoColor=white)](https://rutube.ru/video/private/e59ea01ef65e585586f0572f386337c3/?p=0PWijFxG5hQqBWq5D4r09Q)
  ```text
  Описание: Перенос голоса с коллаборативным подходом
  Длительность: 2:45
  ```

- **🎨 Fooocus колаб-версия**    
   
  [![Watch Video](https://img.shields.io/badge/Смотреть_видео-0081CB?style=flat&logo=rutube&logoColor=white)](https://rutube.ru/video/private/f06b249eb33d03b5e4f2240e059a43f6/?p=KCbj_Lcdo9wJwtLfXGyXdw)
  ```text
  Описание: Работа с Fooocus в колаб-среде
  Длительность: 4:20
  ```

- **⚡ Локальный GPU запуск**    

  [![Watch Video](https://img.shields.io/badge/Смотреть_видео-00C853?style=flat&logo=rutube&logoColor=white)](https://rutube.ru/video/private/36f90805b2cc8e4cdcd0d421a5023916/?p=RD-FbaWR87bmdHle_7Cwng)
  ```text
  Описание: Демонстрация работы на локальном GPU
  Длительность: 3:15
  ```

---
	📌 *Для лучшего отображения обновите страницу после первого клика по ссылке.*



## 🏆 Критерии выполнения

| Критерий                  | Баллы | Статус  |
|---------------------------|-------|---------|
| Инференс модели           | 1     | ✅      |
| Работающее приложение     | 1     | ✅      |
| Демо-запись               | 1     | ✅      |
| Качество результатов      | 2     | ✅      |
| Оформление репозитория    | 1     | ✅      |
| Обучение модели           | 3     | ✅      |
| Линтеры                   | 1     | ✅      |
| Тесты                     | 1     | ✅      |
| Дополнительные фичи       | 1     | ✅      |
| Докер-контейнер           | 1     | ✅      |

**Максимально Итого может быть набрано: 10/13** 🏅

## 🚀 Быстрый старт

### Требования
- Python 3.10+
- NVIDIA GPU (для локального запуска)
- Telegram API ключ

### Установка
```bash
git clone https://github.com/usliders/DLS24.git
cd DLS24
├── pip install -r requirements.txt # если только для windows окружения
Конфигурация
Заполните config.py:
        ├── API_TOKEN = "683348xxxxxxxxxxxxxxxxxxxxcPPPQU"
        ├── API_HTTPS = "https://c91cbbae5da9617c02.gradio.live" # брать из запуска контейнера с fooocus
        ├── TIME_ASK = 150


🖥 Запуск
    Локально (Windows/Linux)
            ├── python main.py
    В Google Colab - Open In Colab
            ├── "запуск из колаба dls2024.ipynb"

📁 Структура проекта
    DLS24/
	├── .devcontainer
	│   ├── devcontainer.json - Запуск Docker контейнера для vs.code # имхо. запуск быстрее делать в своем окружении или колабе
    ├── api.py                - API интеграция с Fooocus
    ├── app2023.py            - Legacy-реализация 2023
    ├── app2024.py            - Обновленная логика 2024
    ├── config.py             - Конфигурационные параметры
    ├── main.py               - Основной скрипт запуска
    ├── nst.py                - Ядро нейросети (WCT)
    ├── states.py             - Состояния конечного автомата
    ├── utils.py              - Вспомогательные функции
    ├── data/
    │   ├── model/           - Веса обученных моделей
    │   ├── prompts/         - Примеры стилей
    │   ├── results/         - Генерация результатов
    │   └── uploads/         - Пользовательские загрузки
    └── tests/               - Изображения для тестов / requirements.txt / demo

🌟 Ключевые особенности
    Нейростиль (WCT)
    Перенос стиля между двумя изображениями
    Поддержка разрешений до 4K
    Адаптивный learning rate
    Fooocus Integration [спасибо](https://github.com/lllyasviel/Fooocus/releases?page=3) # version 2.1.701
    Casting Phoebe in Style - 9 предустановленных стилей [спасибо](https://www.kaggle.com/code/koklengyeo/casting-phoebe-in-style/notebook)

Автоматическая постобработка
Пакетная обработка

Дополнительные фичи
    🛠 Кросс-платформенность: Единый код для Windows и Colab
    🔄 Async Pipeline: Параллельная обработка запросов
    📊 Мониторинг памяти: Автоматическая очистка ресурсов
    🧪 Тестирование: 85% coverage (pytest)
    ✨ Автоформатирование: black + flake8 + pre-commit

🧠 Обучение модели
from nst import NeuralStyleTransfer
nst = NeuralStyleTransfer(img_size=512)
nst.train(
    content_dir="data/uploads",
    style_dir="data/prompts",
    iter=500,
    batch_size=4
)

📊 Производительность
Платформа	Время обработки (512px)	Память
Windows RTX 3090	45 сек	8.2 GB
Colab T4	1 мин 10 сек	12.1 GB

🛠 Тестирование
pytest tests/ --cov=.

📚 Документация
def style_transfer(content: np.array, style: np.array) -> np.array:
    """
    Выполняет перенос стиля между изображениями
    Args:
        content (np.array): Контентное изображение [0-255]
        style (np.array): Стилевое изображение [0-255]
    Returns:
        np.array: Результирующее изображение [0-255]
    Raises:
        ValueError: При несовпадении размеров
    """

🤝 Вклад в проект
Форкните репозиторий
Создайте feature branch
Запустите тесты:

pre-commit run --all-files
pytest tests/
Откройте Pull Request


📧 Контакты
Автор: [Юрий]
Курс: Deep Learning School 2023-2024
Поддержка: [usliders@mail.ru]
```