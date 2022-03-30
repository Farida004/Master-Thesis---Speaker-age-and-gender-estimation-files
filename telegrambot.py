import telebot

TOKEN = '5001828085:AAEdwreEZqNRmNWbsU9LQhDzil_3E2qM3UQ'

bot=telebot.TeleBot(TOKEN)
user_dict = {}

class User:
    def __init__(self, name):
        self.name = name

@bot.message_handler(commands=['start'])
def start_message(message): 
    sent = bot.send_message(message.chat.id, 'Answer the following questions first. What is your age and gender ? Answer in one line')
    bot.register_next_step_handler(sent, hello)

def hello(message):
    open('problem.txt', 'a').write('\n'+ message.text)
    new = bot.send_message(message.chat.id, 'And now I would ask you to record your voice in Azerbaijani. You can introduce yourself.')
    user = User(message.text)
    user_dict[message.chat.id] = user
    bot.register_next_step_handler(new, voice_processing)

@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    user = user_dict[message.chat.id]
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    with open("C:\\Users\\ASUS\\Desktop\\dataforproject\\" +user.name + ".ogg", 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.send_message(message.chat.id, 'Thank you!')

bot.polling(none_stop=True)

