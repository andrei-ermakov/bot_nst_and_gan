import asyncio
from aiogram import filters, types
from aiogram.types import Message
from .for_admin import send_to_admin
from loader import dp, bot
from handler.Class_NST import NeuralTransferStyle
from handler.Class_horse2zebra import GAN_for_Transform

# для обработки событий о поступающих фото
dict_for_transfer_style = {}
dict_for_h2z = {}
dict_for_z2h = {}

text_instruction = "\n" \
                   "-> Для того, чтобы перезапустить бота, нажмите /start.\n" \
                   "-> Для того, чтобы увидеть возможности бота, нажмите /show_what_you_can.\n" \
                   "-> Для того, чтобы выполнить перенос стиля одной картинки на другую, нажмите /go.\n" \
                   "-> Для того, чтобы превратить лошадь в зебру, нажмите /go_horse2zebra.\n" \
                   "-> Для того, чтобы превратить зебру в лошадь, нажмите /go_zebra2horse.\n"


# запускается при запуске бота
@dp.message_handler(filters.CommandStart())
async def send_welcome(message: Message):
    await send_to_admin(dp)
    await message.answer("Привет, Вы запустили бот, который умеет переносить стиль одной картинки"
                         " на другую с помощью алгоритма машинного обучения, а также превращать лошадей в зебр"
                         ", а зебр - в лошадей. Можете попробовать это сами, следуя инструкциям ниже.\n" + text_instruction)
    dict_for_transfer_style[0] = None
    dict_for_h2z[0] = None
    dict_for_z2h[0] = None


# обрабатывает команду /show_what_you_can
@dp.message_handler(filters.Command(commands=['show_what_you_can']))
async def show_what_you_can(message: Message):
    # первое сообщение пользователю:
    await bot.send_message(chat_id=message.from_user.id, text="Вот, что я могу:")

    # Create media group
    media_content, media_style, media_finish = types.MediaGroup(), types.MediaGroup(), types.MediaGroup()
    media_h2z, media_z2h = types.MediaGroup(), types.MediaGroup()
    media_content.attach_photo(types.InputFile(r"image_for_bot/karabash.jpg"))
    media_style.attach_photo(types.InputFile(r"image_for_bot/shanghai.jpg"))
    media_finish.attach_photo(types.InputFile(r"image_for_bot/result_nst.jpg"))
    media_h2z.attach_photo(types.InputFile(r"image_for_bot/horse2zebra.jpg"))
    media_z2h.attach_photo(types.InputFile(r"image_for_bot/zebra2horse.jpg"))

    # Wait a little...
    await asyncio.sleep(1)
    # отправляем картинку-основу
    await bot.send_message(chat_id=message.from_user.id, text="Я умею переносить стиль одной картинки на другую. Приведу простой пример"
                                                              "\nВот картинка-основа \nОжидайте...")
    # для того, чтобы ползунок вверху писал: "Отправляет фото...>>>"
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=message.from_user.id, media=media_content)

    # отправляем картинку-стиль
    await asyncio.sleep(4)
    await bot.send_message(chat_id=message.from_user.id, text="Вот картинка-стиль \nОжидайте...")
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=message.from_user.id, media=media_style)

    # отправляем картинку-результат
    await asyncio.sleep(2)
    await bot.send_message(chat_id=message.from_user.id, text="А вот то, что мы можем получить \n")
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=message.from_user.id, media=media_finish)

    # отправляем преобразование лошадь-зебра
    await asyncio.sleep(4)
    await bot.send_message(chat_id=message.from_user.id, text="Также умею превращать лошадей в зебр...")
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=message.from_user.id, media=media_h2z)

    # отправляем преобразование зебра-лошадь
    await asyncio.sleep(4)
    await bot.send_message(chat_id=message.from_user.id, text="... и зебр в лошадей...")
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=message.from_user.id, media=media_z2h)

    await asyncio.sleep(4)
    await bot.send_message(chat_id=message.from_user.id, text=text_instruction)


# функция для отработки команды /go
@dp.message_handler(filters.Command(commands=['go']))
async def lets_go_work(message: Message):
    user_id = message.from_user.id
    await bot.send_message(chat_id=message.from_user.id, text="Загрузите поочередно Ваши картинки (основа и стиль)."
                                                              "\nПроцесс обработки картинок займет примерно 5-7 минут\n"
                                                              "\nЗагрузите картинку-основу.")
    dict_for_transfer_style[0] = True
    dict_for_h2z[0] = None
    dict_for_z2h[0] = None
    dict_for_transfer_style[user_id] = []


# функция для отработки команды /go_horse2zebra
@dp.message_handler(filters.Command(commands=['go_horse2zebra']))
async def lets_go_h2z(message: Message):
    user_id = message.from_user.id
    await bot.send_message(chat_id=message.from_user.id, text="Загрузите картинку лошади")
    dict_for_h2z[0] = True
    dict_for_transfer_style[0] = None
    dict_for_z2h[0] = None
    dict_for_h2z[user_id] = []


# функция для отработки команды /go_zebra2horse
@dp.message_handler(filters.Command(commands=['go_zebra2horse']))
async def lets_go_z2h(message: Message):
    user_id = message.from_user.id
    await bot.send_message(chat_id=message.from_user.id, text="Загрузите картинку зебры")
    dict_for_z2h[0] = True
    dict_for_transfer_style[0] = None
    dict_for_h2z[0] = None
    dict_for_z2h[user_id] = []


# функция для отработки команды /get_picture
@dp.message_handler(filters.Command(commands=['get_picture']))
async def lets_go_work(message: Message):
    user_id = message.from_user.id

    # создаем экземпляр класса types.MediaGroup() для добавления картинки-основы
    # file["file_id"] - загруженная на сервер Telegram картинка, поэтому мы можем вывести ее таким образом
    media_content, media_style = types.MediaGroup(), types.MediaGroup()
    media_content.attach_photo(dict_for_transfer_style[user_id][0]["file_id"])
    media_style.attach_photo(dict_for_transfer_style[user_id][1]["file_id"])

    # блок показа пользователю введенных им данных
    await bot.send_message(chat_id=message.from_user.id, text="Я начал обработку. \n"
                                                              "Вот что вы загрузили...")
    await asyncio.sleep(1)
    await bot.send_message(chat_id=user_id, text="Ваша картинка-основа")
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=user_id, media=media_content)
    await bot.send_message(chat_id=user_id, text="Ваша картинка-стиль")
    await asyncio.sleep(1)
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=user_id, media=media_style)
    await bot.send_message(chat_id=user_id, text="Пожалуйста ожидайте 5-7 минут")

    # блок работы NST
    image_content = await bot.download_file(dict_for_transfer_style[user_id][0].file_path)
    image_style = await bot.download_file(dict_for_transfer_style[user_id][1].file_path)
    nst = NeuralTransferStyle(content_picture=image_content,
                              style_picture=image_style,
                              max_size=400,
                              number_epochs=700).train_nst()
    media_result = types.MediaGroup()
    media_result.attach_photo(nst)
    await bot.send_message(chat_id=user_id, text="Вот что получилось...")
    await asyncio.sleep(1)
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=user_id, media=media_result)
    await bot.send_message(chat_id=user_id, text="Надеюсь, Вам понравилось. Вы можете попробовать снова!"+text_instruction)
    dict_for_transfer_style[0] = None
    dict_for_transfer_style[user_id] = []


# функция для отработки команды /get_picture_h2z
@dp.message_handler(filters.Command(commands=['get_picture_h2z']))
async def h2z(message: Message):
    user_id = message.from_user.id

    # блок работы GAN_h2z
    horse = await bot.download_file(dict_for_h2z[user_id][0].file_path)
    gan = GAN_for_Transform(horse, h2z=True).transformation()
    media_result = types.MediaGroup()
    media_result.attach_photo(gan)
    await bot.send_message(chat_id=user_id, text="Вот что получилось...")
    await asyncio.sleep(1)
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=user_id, media=media_result)
    await bot.send_message(chat_id=user_id, text="Надеюсь, Вам понравилось. Вы можете попробовать снова!"+text_instruction)
    dict_for_h2z[0] = None
    dict_for_h2z[user_id] = []


# функция для отработки команды /get_picture_z2h
@dp.message_handler(filters.Command(commands=['get_picture_z2h']))
async def z2h(message: Message):
    user_id = message.from_user.id

    # блок работы GAN_z2h
    horse = await bot.download_file(dict_for_z2h[user_id][0].file_path)
    gan = GAN_for_Transform(horse, h2z=False).transformation()
    media_result = types.MediaGroup()
    media_result.attach_photo(gan)
    await bot.send_message(chat_id=user_id, text="Вот что получилось...")
    await asyncio.sleep(1)
    await types.ChatActions.upload_photo()
    await bot.send_media_group(chat_id=user_id, media=media_result)
    await bot.send_message(chat_id=user_id, text="Надеюсь, Вам понравилось. Вы можете попробовать снова!"+text_instruction)
    dict_for_z2h[0] = None
    dict_for_z2h[user_id] = []


# если пользователь просто пишет текст, бот вернет его ему
@dp.message_handler()
async def echo(message: Message):
    text = f"Вы написали: {message.text}"
    await bot.send_message(chat_id=message.from_user.id,
                           text=text + text_instruction)


# функция для загрузки картинок
@dp.message_handler(content_types="photo")
async def scan_message(message: Message):
    # Блок загрузки картинок для наложения стилей
    user_id = message.from_user.id

    # получаем id нашей картинки
    photo_id = message.photo[0].file_id
    # получаем файл
    file = await bot.get_file(photo_id)

    # если пользователь не выбрал режим, ему придет сообщение
    if dict_for_transfer_style[0] is None and dict_for_h2z[0] is None and dict_for_z2h[0] is None:
        await bot.send_message(chat_id=user_id, text="Выберете режим работы бота, а потом загружайте картинки\n"+text_instruction,
                               allow_sending_without_reply=True)

    # определяем с чем работаем (зебра-лошадь, лошадь-зебра, наложение стилей)
    if dict_for_transfer_style[0] is not None:
        # проверяем, есть ли такой ключ в словаре. Если нет - создаем новый
        if len(dict_for_transfer_style[user_id]) == 0:
            dict_for_transfer_style[user_id].append(file)
            await bot.send_message(chat_id=user_id, text="Картинка-основа загружена. Загрузите картинку стиль...",
                                   allow_sending_without_reply=True)
        else:
            if len(dict_for_transfer_style[user_id]) == 1:
                dict_for_transfer_style[user_id].append(file)
                # allow_sending_without_reply=True - выводит надпись всего один раз
                await bot.send_message(chat_id=user_id, text="Картинка-стиль загружена."
                                                             "Для начала обработки нажмите /get_picture",
                                       allow_sending_without_reply=True)

    # Блок загрузки картинки зебры или лошади
    if dict_for_h2z[0] is not None:
        if len(dict_for_h2z[user_id]) == 0:
            dict_for_h2z[user_id].append(file)
            await bot.send_message(chat_id=user_id, text="Картинка лошади загружена. Для начала обработки нажмите /get_picture_h2z",
                                   allow_sending_without_reply=True)

    if dict_for_z2h[0] is not None:
        if len(dict_for_z2h[user_id]) == 0:
            dict_for_z2h[user_id].append(file)
            await bot.send_message(chat_id=user_id, text="Картинка зебры загружена. Для начала обработки нажмите /get_picture_z2h",
                                   allow_sending_without_reply=True)