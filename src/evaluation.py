import tensorflow as tf
import numpy as np

from preprocessing import *
from utils import shuffle, get_batch_data

MAX_SENTENCE_LEN = 10
EMBEDDING_DIM = 100
EMBEDDING_FILE_PATH = 'data/embeddings.txt'


def load_model( model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def evaluate(model , word2Idx_path, embeddings_path ,test_path , test_preprocessed_path , results_path ):
    data = load_dataframe( test_path )
    text_corpus , aspect_corpus = process_data(data , mode = "test")
    preprocessed_file(test_preprocessed_path , text_corpus , aspect_corpus , labels=None)
    
    word2Idx = load_word2Idx(word2Idx_path)
    embedding_matrix = load_embedding_matrix(embeddings_path , word2Idx)
    x , x_r , target_words = load_inputs(test_preprocessed_path ,word2Idx,embedding_matrix, MAX_SENTENCE_LEN , mode = "test" )
    x , x_r , target_words = final_finish_processing(x , x_r , target_words)
    inputs_fw , inputs_bw  = prepare(x , x_r, target_words, MAX_SENTENCE_LEN ,EMBEDDING_DIM, tr_y = None )
    
    pred = model.predict( (inputs_fw , inputs_bw) )
    pred = tf.math.argmax(pred, axis=1)
    pred = pred.numpy()
    data["label"] = pred
    data.to_csv(results_path)

model = load_model('models/saved_models/trained_model.h5' )
# Be sure to create preprocessed_test.txt file before runing this below line
evaluate(model , 'data/word2Idx.txt' , 'data/embeddings.txt' ,'data/test.csv' ,'data/preprocessed_test.txt' ,'data/results/results.csv')