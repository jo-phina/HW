from config import API_TOKEN
import telebot

bot = telebot.TeleBot(token=API_TOKEN)  # object referring to my bot in telegram

@bot.message_handler(commands=['start'])  # handles messages. whatever function comes afterwards will be filtered

def welcome(message):
    welcome_text = f'Hello {message.from_user.first_name}, welcome to JoBot!'
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(func=lambda message:True)

def reply_function(message):
    bot.reply_to(message, text="Your has been lost in the depths of the internet.")


bot.polling()