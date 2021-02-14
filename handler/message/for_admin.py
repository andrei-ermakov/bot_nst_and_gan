from loader import bot
from data import config


async def send_to_admin(Dispatcher):
    await bot.send_message(chat_id=config.admin_id, text="Бот запущен")