import telebot
from telebot import types
from ultralytics import YOLO
import cv2
from dotenv import load_dotenv
import os
from PIL import Image
import time
import db
import io
from typing import Dict, List, Optional, Any

load_dotenv()

BOT_API_TOKEN: Optional[str] = os.getenv("BOT_API_TOKEN")
if BOT_API_TOKEN is None:
    raise ValueError("BOT_API_TOKEN not found in environment variables.")

bot = telebot.TeleBot(BOT_API_TOKEN)
db_manager: db.DatabaseManager = db.DatabaseManager("mydatabase.db")

model = YOLO("models/yolov8x.pt")


@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Привет! Отправь мне фото с коровами, и я посчитаю их количество и отмечу на изображении.",
    )


@bot.message_handler(commands=["stats"])
def send_stats(message: types.Message):
    stats = db_manager.get_total_statistics()

    response = [
        "📊 Общая статистика:",
        f"• Всего запросов: {stats['total_requests']}",
        f"• Общее время обработки: {stats['total_processing_time']:.2f} сек",
        f"• Всего обнаружено объектов: {stats['total_objects']}",
        "\nТоп пользователей:",
    ]

    for i, user in enumerate(stats["users_stats"][:5], 1):
        response.append(
            f"{i}. ID {user['user_id']}: "
            f"{user['requests']} запросов, "
            f"{user['objects']} объектов"
        )

    bot.send_message(message.chat.id, "\n".join(response))


@bot.message_handler(commands=["history"])
def send_welcome(message: types.Message):
    try:
        history = db_manager.get_user_history(message.from_user.id)

        if not history:
            bot.reply_to(message, "📭 Ваша история запросов пуста")
            return

        response = ["📖 Ваша история запросов:\n"]

        for i, record in enumerate(history[:10], 1):
            date = record[4]
            response.append(
                f"{i}. {date.split('.')[0]}\n"
                f"   ⚙️ Обработка: {record[2]:.2f} сек\n"
                f"   🎯 Объектов: {record[3]}\n"
            )

        total_requests = len(history)
        total_objects = sum(record[3] for record in history)
        total_time = sum(record[2] for record in history)

        response.append(
            "\n📊 Итого:\n"
            f"• Всего запросов: {total_requests}\n"
            f"• Общее время обработки: {total_time:.2f} сек\n"
            f"• Всего объектов: {total_objects}"
        )

        bot.send_message(message.chat.id, "".join(response), parse_mode="Markdown")

    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка при получении истории: {str(e)}")


@bot.message_handler(content_types=["photo"])
def handle_photo(message: types.Message):
    try:
        user_id = message.chat.id
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        new_img = f"{time.time()}_{user_id}.jpg"
        file_path = f"secret/{new_img}"

        with open(file_path, "wb") as new_file:
            new_file.write(downloaded_file)

        process_start = time.time()
        img = cv2.imread(file_path)
        results = model.predict(img, conf=0.6)
        processing_time = time.time() - process_start

        annotated_img = results[0].plot()

        class_counts = {}
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        total_objects = sum(class_counts.values())

        caption_parts = [f"Найдено объектов: {total_objects}"]
        for class_name, count in class_counts.items():
            caption_parts.append(f"{class_name.capitalize()}: {count}")
        caption = "\n".join(caption_parts)

        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_img)

        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="JPEG")
        img_byte_arr.seek(0)

        bot.send_photo(message.chat.id, img_byte_arr, caption=caption)

        db_manager.add_record(
            user_id=user_id,
            image_processing_time=processing_time,
            objects_count=total_objects,
        )

    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {str(e)}")


bot.polling(none_stop=True)
