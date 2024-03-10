import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import sys
sys.path.append('C:\\Users\\HP\\anaconda3\\pkgs\\torchaudio-2.1.2-py311_cu121\\info\\test\\test\\torchaudio_unittest\\backend\\dispatcher')
# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')
# Setup the sidebar
with st.sidebar: 
    st.image('https://img.freepik.com/free-vector/colorful-wavy-background_23-2148493462.jpg?w=900&t=st=1710080852~exp=1710081452~hmac=540c92f1b6be87aea562b954aa66c9a1d84d18f03f6dba14ddf6d9573dd479e5')
    st.title('Speak')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Lip Reading App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)
if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

        with col2: 
            # st.info('This is all the machine learning model sees when making a prediction')
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            # imageio.mimsave('animation.gif', video, duration=len(video)/100)
            # st.image('animation.gif', width=400) 

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            
            st.text(converted_prediction)
                