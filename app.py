## Import Libraries
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation
from tensorflow.keras.optimizers import RMSprop

# Data
filepath=tf.keras.utils.get_file('shakesphere.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()
#selecting a part of text for fast training
text=text[300000:800000]

#Data preprocessing
characters=sorted(set(text)) #all the unique characters present in text
char_to_index=dict((c,i) for i,c in enumerate(characters))
index_to_char=dict((i,c) for i,c in enumerate(characters))

#Feature and Target
Sequence_len=40 #we are going to use 40 characters in order to predict next character
step_size=3
sentences=[] #feature data
next_characters=[] #target data
for i in range(0,len(text)-Sequence_len,step_size):
    sentences.append(text[i:i+Sequence_len])
    next_characters.append(text[i+Sequence_len])

#Categorical to Numerical Data
x=np.zeros((len(sentences),Sequence_len,len(characters)),dtype=bool) #input
y=np.zeros((len(sentences),len(characters)),dtype=bool) #target
for i,sentence in enumerate(sentences):
    for t,character in enumerate(sentence):
        x[i,t,char_to_index[character]]=1
    y[i,char_to_index[next_characters[i]]]=1

#Neural Network
model=Sequential()
model.add(LSTM(128,input_shape=(Sequence_len,len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.01))

#Train Model
model.fit(x,y,batch_size=256,epochs=4)
model.save('text_generator.keras') #instead of training model again and again we can now load this saved model