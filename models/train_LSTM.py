import tensorflow as tf
import numpy as np
from preprocessing import *
from utils import *


EPOCHS = 100
MAX_SENTENCE_LEN = 10
EMBEDDING_DIM = 100
BATCH_SIZE = 64

embedding_matrix = {} 
word2Idx  = {}

def model(input_fw , inputs_bw,  inputs , hidden):
   
   inputs1 = tf.keras.Input(shape=inputs)
   inputs2 = tf.keras.Input(shape=inputs)

   # forward
   fw = tf.keras.layers.LSTM(hidden)(inputs1)

   #backward
   bw = tf.keras.layers.LSTM(hidden)(inputs2)

   concatenated =  tf.concat([fw, bw], 1)

   x = tf.keras.layers.Dense(512, activation="relu")(concatenated)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(256 , activation="relu")(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(64 , activation="relu")(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(32 , activation="relu")(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(8 , activation="relu")(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   x = tf.keras.layers.Dense(3 , activation="sigmoid")(concatenated)
   x = tf.keras.activations.softmax(x)

   model = tf.keras.Model(inputs=[inputs1 , inputs2], outputs=[x])
   optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.01, epsilon=1e-07)
   model.compile(loss="categorical_crossentropy" , optimizer=optimizer, metrics=["accuracy"])

   return model

def run_model(model ,inputs_fw , inputs_bw , y , epochs=100  ,generator=None , save = True , save_path=None):
    if generator==None:
        r = model.fit((inputs_fw , inputs_bw ), y , epochs = epochs  , steps_per_epoch = len(y)//BATCH_SIZE )
    else:
        r = model.fit_generator(generator=get_batch_data(inputs_fw , inputs_bw ,y , BATCH_SIZE) 
                            
    if save == True:
         tf.keras.models.save_model(model , save_path)
    return r
                                
                                
def run(train_csv_file_path , embedding_file_path):
        data = load_dataframe(train_csv_file_path)
        text_corpus, aspect_corpus , labels = process_data(data , mode="train")
        preprocessed_file("data/preprocessed_train.txt" ,text_corpus ,aspect_corpus , labels)
        embedding_matrix , word2Idx = create_embeddings_matrix(text_corpus , aspect_corpus , embedding_file_path)
        #TODO create a file for storing word2Idx here                    
        x , x_r , y , target_words = load_inputs("data/preprocessed_train.txt" , word2Idx , MAX_SENTENCE_LEN , mode="train")
        x , x_r , y = final_finish_processing(x , x_r , target_words)
        inputs_fw , inputs_bw , y = prepare(x , x_r , y , embedding_matrix , MAX_SENTENCE_LEN , EMBEDDING_DIM , y)
        r = run_model("models/saved_models/trained_model.h5" , inputs_fw , inputs_bw , y ,EPOCHS , None , True , None)
        return r