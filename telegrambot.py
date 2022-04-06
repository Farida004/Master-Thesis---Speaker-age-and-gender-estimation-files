import telebot
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense , Activation , Dropout
import librosa
import IPython.display as ipd 
import subprocess

TOKEN = '5001828085:AAEdwreEZqNRmNWbsU9LQhDzil_3E2qM3UQ'

bot=telebot.TeleBot(TOKEN)
user_dict = {}

class User:
    def __init__(self, name):
        self.name = name

def extract_feature(file_name):
    # load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    y, sr = librosa.load(file_name, mono=True, duration=30)
    
    # get the feature 
    feature = librosa.feature.mfcc(y=y,sr=sample_rate,n_mfcc=50)
    # scale the features
    feature_scaled = np.mean(feature.T,axis=0)
    
    # return the array of features
    return np.array([feature_scaled])

def print_prediction(file_name,model, le):
    
    # extract feature from the function defined above
    prediction_feature = extract_feature(file_name) 
    
    # get the id of label using argmax
    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    
    # get the class label from class id
    predicted_class = le.inverse_transform(predicted_vector)
    
    # display the result
    return ("The predicted class is: " + predicted_class[0])

@bot.message_handler(commands=['start'])
def start_message(message): 
    new = bot.send_message(message.chat.id, 'Record the voice in Azerbaijani so the bot can predict you age and gender!.')
    user = User(message.text)
    user_dict[message.chat.id] = user
    bot.register_next_step_handler(new, voice_processing)


@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    extracted_df_gen = pd.read_pickle("gen_features.pkl")  
    extracted_df_age = pd.read_pickle("age_features.pkl")  
    x = np.array(extracted_df_gen['feature'].tolist())
    y = np.array(extracted_df_gen['class'].tolist())
    le = LabelEncoder()
    # transform each category with it's respected label
    Y = to_categorical(le.fit_transform(y))
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2, random_state = 42)
    print("Number of training samples = ", x_train.shape[0])
    print("Number of testing samples = ",x_test.shape[0])

    x1 = np.array(extracted_df_age['feature'].tolist())
    y1 = np.array(extracted_df_age['class'].tolist())
    le1 = LabelEncoder()
    # transform each category with it's respected label
    Y1 = to_categorical(le1.fit_transform(y1))
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, Y1, test_size=0.2, random_state = 42)
    print("Number of training samples = ", x_train1.shape[0])
    print("Number of testing samples = ",x_test1.shape[0])

    user = user_dict[message.chat.id]
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    model_gender = pickle.load(open('MLP_model.pkl','rb'))
    model_age = pickle.load(open('MLP_model.pkl','rb'))
    with open("C:\\Users\\ASUS\\Desktop\\" + str(message.chat.id) + ".ogg", 'wb') as new_file:
        new_file.write(downloaded_file)
    
    subprocess.call(['ffmpeg', '-i', str(message.chat.id) + ".ogg", str(message.chat.id) + ".wav"], shell = True)

    bot.send_message(message.chat.id, print_prediction(str(message.chat.id) + ".wav", model_gender, le))
    bot.send_message(message.chat.id, print_prediction(str(message.chat.id) + ".wav", model_age, le1))
    bot.send_message(message.chat.id, 'Thank you!')

bot.polling(none_stop=True)


