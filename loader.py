import asyncio
from aiogram import Bot, Dispatcher, types
from data import config

# запускаем потоки
loop = asyncio.get_event_loop()

bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
dp = Dispatcher(bot, loop=loop)
