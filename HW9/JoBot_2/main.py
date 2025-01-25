from config import API_TOKEN
import telebot

bot = telebot.TeleBot(token=API_TOKEN)  # object referring to my bot in telegram

@bot.message_handler(commands=['start'])  # handles messages. whatever function comes afterwards will be filtered

def welcome(message):
    welcome_text = f"Hello {message.from_user.first_name}, welcome to JoBot! I am an confused AI assistant - maybe I can help you, but probably not..."
    bot.send_message(message.chat.id, welcome_text)

@bot.message_handler(content_types=['audio', 'document'])
def handle_audio_doc(message):
    if message.audio:
        bot.reply_to(message, "Why did you send me an audio file??!")
    elif message.document:
        bot.reply_to(message, "You sent me a document... What am I supposed to do with that?")

@bot.message_handler(regexp="2025")
def handle_date(message):
    bot.reply_to(message, "Is it 2025 already?")

@bot.message_handler(func=lambda message:True)

def reply_function(message):
    bot.reply_to(message, text="Your message has been lost in the depths of the internet.")


bot.polling()