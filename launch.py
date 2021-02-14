from aiogram import executor
from handler import dp, send_to_admin


if __name__ == "__main__":
    executor.start_polling(dp, on_startup=send_to_admin, timeout=720)