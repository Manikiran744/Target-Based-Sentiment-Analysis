import numpy as np
import tensorflow as tf
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


MAX_SENTENCE_LEN = 10
EMBEDDING_DIM = 100


def load_dataframe( path ):
    data = pd.read_csv(path)
    return data


def process_data(data , mode = "train"):

    text_corpus = []
    aspect_corpus = []
    labels = []

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    # ps = PorterStemmer()
    for i in range(0 , len(data)):

      review_text = re.sub('[^a-zA-Z]', ' ', data['text'][i])
      review_text = review_text.lower()
      review_text = review_text.split()
      # review_text = [ps.stem(word) for word in review_text if not word in set(all_stopwords)]
      review_text = ' '.join(review_text)
      text_corpus.append(review_text)

      aspect = re.sub('[^a-zA-Z]', ' ', data['aspect'][i])
      aspect = aspect.lower()
      aspect = aspect.split()
      # aspect = [ps.stem(word) for word in aspect if not word in set(all_stopwords)]
      aspect = ' '.join(aspect)
      aspect_corpus.append(aspect)

      if mode == "train":
        labels.append(int(data['label'][i]))

    if mode=="train":
        return text_corpus , aspect_corpus , labels
    else:
        return text_corpus , aspect_corpus
    
def preprocessed_file(filepath , text_corpus , aspect_corpus , labels=None):

    file = open(filepath, 'w')

    sentiment = "$t$"
    
    for i  in range(len(text_corpus)):
        
       string = text_corpus[i]
       string = string.replace(aspect_corpus[i] , sentiment , 1)
       file.writelines(string+'\n')
       file.writelines(aspect_corpus[i]+'\n')
       if labels != None:
           file.writelines(str(labels[i])+'\n')

    file.close()


def create_embeddings_matrix(text_corpus , aspect_corpus , embedding_path_file):
    
    all_wordset = set()

    for data in text_corpus:
       words = data.split(' ')
       for i in words:
           all_wordset.add(i)

    for data in aspect_corpus:
       words = data.split(' ')
       for i in words:
           all_wordset.add(i)

    word2Idx = {}

    if len(word2Idx) == 0:
      word2Idx["PADDING_TOKEN"] = len(word2Idx)
      word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
    for word in all_wordset:
      word2Idx[word] = len(word2Idx)

    embedding_matrix_1= np.zeros((len(word2Idx), EMBEDDING_DIM))

    lines = open(embedding_path_file).readlines()

    for i in range(len(lines)):
       lists = lines[i].split(' ')
       word = lists[0]
       embeddings = lists[1:EMBEDDING_DIM]
       embeddings = embeddings
       embedding_matrix_1[i] = embeddings

    return embedding_matrix_1 , 


def load_word2Idx(path):
    word2Idx = {}
    lines = open(path).readlines()

    for i in range(len(lines)):
       lists = lines[i].split(' ')
       word = lists[0]
       id = lists[1]
       word2Idx[word] = id

    return word2Idx 


def load_embedding_matrix(embeddings_path , word2Idx):
    embedding_matrix_1= np.zeros((len(word2Idx), EMBEDDING_DIM))

    lines = open(embeddings_path).readlines()

    for i in range(len(lines)):
       lists = lines[i].split(' ')
       word = lists[0]
       embeddings = lists[1:EMBEDDING_DIM]
       embeddings = embeddings
       embedding_matrix_1[i] = embeddings

    return embedding_matrix_1 , 


def change_y_to_onehot(y):
    from collections import Counter
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)



def load_inputs(input_file ,word2Idx, MAX_SENTENCE_LEN , mode = "train" ):
    x = []
    x_r = []
    labels = list()

    target_words = []

    lines = open(input_file).readlines()

    for i in range(0, len(lines), 3):
        target_word = lines[i+1].lower().split()
        target_word = map(lambda w: word2Idx.get(w, 0), target_word)
        target_word = list(target_word)
        target_words.append([target_word])
        
        if mode=="train":
            labels.append(lines[i + 2].strip().split()[0])

        words = lines[i].lower().split()
        words_l, words_r = [], []
        flag = True

        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word2Idx:
                    words_l.append(word2Idx[word])
            else:
                if word in word2Idx:
                    words_r.append(word2Idx[word])

        words_l.extend(target_word)
    #     sen_len.append(len(words_l))
        x.append(words_l + [0] * (MAX_SENTENCE_LEN - len(words_l)))

        tmp = list(target_word) + words_r
        tmp.reverse()
    #     sen_len_r.append(len(tmp))
        x_r.append(tmp + [0] * (MAX_SENTENCE_LEN - len(tmp)))
    
    if mode == "train":
        y = np.array(change_y_to_onehot(labels))
    if mode=="train":
        return np.asarray(x), np.asarray(x_r), np.asarray(y), np.asarray(target_words)
    elif mode=="test":
         return np.asarray(x),np.asarray(x_r),np.asarray(target_words)
        
        
def final_finish_processing(tr_x , tr_x_bw , tr_target_word):
    tr_x_list = []
    for i in tr_x:
        tr_x_list.append(i)
                         
    data = pd.DataFrame(tr_x_list)
    tr_x_list = data.iloc[: , :MAX_SENTENCE_LEN ].values

    
    tr_x_bw_list = []
    for i in tr_x_bw:
        tr_x_bw_list.append(i)
        
    data = pd.DataFrame(tr_x_bw_list)
    data.head()
    tr_x_bw_list = data.iloc[: , :MAX_SENTENCE_LEN ].values
        

    tr_target_word_list = []
    for i in tr_target_word:
        helper = []
        for j in i:
            helper.append(j)
        tr_target_word_list.append(helper)
        
    target_words = []

    for i in tr_target_word_list:
        target_words.append(*i) 


    data = pd.DataFrame(target_words)
    data = data.replace(np.nan, 0)

    target_words = data.iloc[: ,:MAX_SENTENCE_LEN].values
    
    return tr_x_list , tr_x_bw_list , target_words


def prepare(x , x_bw, target_words, embedding_matrix_1 ,MAX_SENTENCE_LEN , EMBEDDING_DIM ,tr_y = None  ):
    
        x = tf.convert_to_tensor(np.array(x))
        inputs_fw = tf.nn.embedding_lookup(embedding_matrix_1, x)
      
        x_bw = tf.convert_to_tensor(np.array(x_bw))
        inputs_bw = tf.nn.embedding_lookup(embedding_matrix_1, x_bw)
        
        
        target_words = tf.convert_to_tensor(np.array(target_words) , dtype="int64")
        
        target = tf.reduce_mean(tf.nn.embedding_lookup(embedding_matrix_1,target_words), 1, keepdims=True)
        
     

        batch_size = tf.shape(inputs_bw)[0]
        target = tf.repeat(target, [ MAX_SENTENCE_LEN ], axis=1)
        inputs_fw = tf.concat([inputs_fw, target], 2)
        inputs_bw = tf.concat([inputs_bw, target], 2) 
        
        if tr_y != None:
            
            y = tf.convert_to_tensor(np.array(tr_y))
            return inputs_fw , inputs_bw , y
        else:
            return inputs_fw , inputs_bw
        
        

    
        