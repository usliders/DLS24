# states.py
from aiogram.dispatcher.filters.state import State, StatesGroup

# Объявления статусов
class StyleChoice(StatesGroup):
    ChoosingStyle = State()
    ChoosingNextStep = State()

class DialogStates(StatesGroup):
    Start = State()
    ChoosingVariant = State()
    WaitingForPhotos = State()
    WaitingForPhoto = State()
    WaitingForContentPhoto = State()
    WaitingForStylePhoto = State()
    WaitingForAlpha = State()
    Finish = State()
    Continue = State()

    