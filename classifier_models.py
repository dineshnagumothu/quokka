import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers

import tensorflow as tf
import tensorflow_hub as hub
print("tensorflow version : ", tf.__version__)
print("tensorflow_hub version : ", hub.__version__)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample

MAX_NB_WORDS = 50000
vocabulary_size=MAX_NB_WORDS
EMBEDDING_DIM = 300

TOPICS_LEN = 100
TEXT_LEN = 1000
ENTITIES_LEN = 1000
TRIPLES_LEN = 1000

def model_making(count, embedding_matrix, sents=False, topics=False, entities=False, triples=False, text=False, fine_tune=False):
  learning_rate = 2e-5
  mod_out=[]
  mod_in=[]
  dropout_rate = 0.3
  if (text==True):
    input_text = tf.keras.layers.Input(shape=(TEXT_LEN,), name='input_text')
    m1_layers = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=fine_tune, name='glove_text_embedding')(input_text)
    m1_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_text_3')(m1_layers)
    m1_layers = tf.keras.layers.Flatten(name='flatten_text')(m1_layers)
    m1_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_text_4')(m1_layers)
    m1_layers = tf.keras.layers.Dense(512, activation='relu', name='dropout_multi_text_5')(m1_layers)
    if(count==1):
      m1_layers = tf.keras.layers.Dropout(dropout_rate)(m1_layers)
      m1_layers = tf.keras.layers.Dense(256,activation='relu', name='dense_1_text')(m1_layers)
      #m1_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_text_1')(m1_layers)
      #m1_layers = tf.keras.layers.Dense(128,activation='relu', name='dense_2_text')(m1_layers)
      m1_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_text_2')(m1_layers)
      m1_layers = tf.keras.layers.Dense(64,activation='relu', name='dense_3_text')(m1_layers)
      m1_layers = tf.keras.layers.Dense(2, activation='softmax', name='dense_output')(m1_layers)
    model_1 = tf.keras.models.Model(inputs=input_text, outputs=m1_layers, name='texts_model')      
    mod_out.append(model_1.output)
    mod_in.append(input_text)

  if(sents==True):
    input_sents = tf.keras.layers.Input(shape=(4096,),name="input_sents")
    #m1_layers = tf.keras.layers.Dense(1024, activation='relu')(input_sents)
    m1_layers = tf.keras.layers.Dropout(dropout_rate)(input_sents)
    m1_layers = tf.keras.layers.Dense(512, activation='relu', name='dense_1_sents')(m1_layers)
    #m1_layers = tf.keras.layers.Dense(32, activation="relu")(m1_layers)
    #m1_layers = tf.keras.layers.Dropout(0.2)(m1_layers)
    if(count==1):
      m1_layers = tf.keras.layers.Dropout(dropout_rate)(m1_layers)
      m1_layers = tf.keras.layers.Dense(256, activation='relu')(m1_layers)
      m1_layers = tf.keras.layers.Dropout(dropout_rate)(m1_layers)
      m1_layers = tf.keras.layers.Dense(128, activation='relu')(m1_layers)
      m1_layers = tf.keras.layers.Dropout(dropout_rate)(m1_layers)
      m1_layers = tf.keras.layers.Dense(64, activation="relu")(m1_layers)
      #m1_layers = tf.keras.layers.Dropout(0.2)(m1_layers)
      m1_layers = tf.keras.layers.Dense(2, activation='softmax', name='dense_output')(m1_layers)
    model_1 = tf.keras.models.Model(inputs=input_sents, outputs=m1_layers, name='sents_model')      
    mod_out.append(model_1.output)
    mod_in.append(input_sents)
  if topics==True:
    input_topics = tf.keras.layers.Input(shape=(TOPICS_LEN,), name='input_topics')
    m2_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_topic_4')(input_topics)
    m2_layers = tf.keras.layers.Dense(512,activation='relu', name='dense_1_topics')(m2_layers)
    if(count==1):
      m2_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_topic_3')(m2_layers)
      m2_layers = tf.keras.layers.Dense(256,activation='relu', name='dense_4_topics')(m2_layers)
      #m2_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_topic_1')(m2_layers)
      #m2_layers = tf.keras.layers.Dense(128,activation='relu', name='dense_2_topics')(m2_layers)
      m2_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_topic_2')(m2_layers)
      m2_layers = tf.keras.layers.Dense(64,activation='relu', name='dense_3_topics')(m2_layers)
      m2_layers = tf.keras.layers.Dense(2, activation='softmax', name='dense_output')(m2_layers)
    model_2 = tf.keras.models.Model(inputs=input_topics, outputs=m2_layers, name='topics_model')
    mod_out.append(model_2.output)
    mod_in.append(input_topics)
  if entities==True:
    input_entities = tf.keras.layers.Input(shape=(ENTITIES_LEN,), name='input_entities')
    m3_layers = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=ENTITIES_LEN, weights=[embedding_matrix], trainable=fine_tune, name='glove_entity_embedding')(input_entities)        
    m3_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_1_entities')(m3_layers)
    m3_layers = tf.keras.layers.Flatten(name='flatten_entities')(m3_layers)
    m3_layers = tf.keras.layers.Dropout(dropout_rate)(m3_layers)
    m3_layers = tf.keras.layers.Dense(512,activation='relu', name='dense_3_entities_2')(m3_layers)
    if(count==1):
      m3_layers = tf.keras.layers.Dropout(dropout_rate)(m3_layers)
      m3_layers = tf.keras.layers.Dense(256,activation='relu', name='dense_3_entities_4')(m3_layers)
      #m3_layers = tf.keras.layers.Dropout(dropout_rate)(m3_layers)
      #m3_layers = tf.keras.layers.Dense(128,activation='relu', name='dense_3_entities_5')(m3_layers)
      m3_layers = tf.keras.layers.Dropout(dropout_rate)(m3_layers)
      m3_layers = tf.keras.layers.Dense(64,activation='relu', name='dense_3_entities_3')(m3_layers)
      m3_layers = tf.keras.layers.Dense(2, activation='softmax', name='dense_output')(m3_layers)
    model_3 = tf.keras.models.Model(inputs=input_entities, outputs=m3_layers)
    mod_out.append(model_3.output)
    mod_in.append(input_entities)
  if triples==True:
    input_triples = tf.keras.layers.Input(shape=(TRIPLES_LEN,), name='input_triples')
    m4_layers = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=fine_tune, name='glove_triple_embedding')(input_triples)
    m4_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_1_triples')(m4_layers)
    m4_layers = tf.keras.layers.Flatten(name='flatten_triples')(m4_layers)
    m4_layers = tf.keras.layers.Dropout(dropout_rate, name='droput_1_triples')(m4_layers)
    m4_layers = tf.keras.layers.Dense(512,activation='relu', name='dense_4_triples_1')(m4_layers)
    if(count==1):
      m4_layers = tf.keras.layers.Dropout(dropout_rate)(m4_layers)
      m4_layers = tf.keras.layers.Dense(256,activation='relu', name='dense_4_triples_4')(m4_layers)
      #m4_layers = tf.keras.layers.Dropout(dropout_rate)(m4_layers)
      #m4_layers = tf.keras.layers.Dense(128,activation='relu', name='dense_4_triples_2')(m4_layers)
      m4_layers = tf.keras.layers.Dropout(dropout_rate)(m4_layers)
      m4_layers = tf.keras.layers.Dense(64,activation='relu', name='dense_4_triples_3')(m4_layers)
      m4_layers = tf.keras.layers.Dense(2, activation='softmax', name='dense_output')(m4_layers)
    model_4 = tf.keras.models.Model(inputs=input_triples, outputs=m4_layers)
    mod_out.append(model_4.output)
    mod_in.append(input_triples)
  
  if (count>1):
    model_cat = tf.keras.layers.concatenate(mod_out)
    model_cat = tf.keras.layers.Dense(512,activation='relu', name='dense_1_cat')(model_cat)
    #model_cat = tf.keras.layers.Dropout(dropout_rate)(model_cat)
    #model_cat = tf.keras.layers.Dense(256,activation='relu', name='dense_2_cat')(model_cat)
    model_cat = tf.keras.layers.Dropout(dropout_rate)(model_cat)
    model_cat = tf.keras.layers.Dense(128,activation='relu', name='dense_3_cat')(model_cat)
    model_cat = tf.keras.layers.Dropout(dropout_rate)(model_cat)
    model_cat = tf.keras.layers.Dense(64, activation="relu", name='dense_out')(model_cat)
    model_cat = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(model_cat)
    model = tf.keras.models.Model(mod_in, model_cat, name='Model_Multi')
  else:
    if sents==True:
      model=model_1
    if text==True:
      model=model_1
    if topics==True:
      model=model_2 
    if entities==True:
      model=model_3
    if triples==True:
      model=model_4
 
  optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
  #ce = tf.keras.losses.BinaryCrossentropy()
  ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


  model.compile(loss=ce, optimizer=optimiser, metrics=['accuracy'])

  #model.summary()
  return model

def compute_metrics(model, X_test, Y_test):
    yhat_probs = model.predict(X_test, verbose=0)
    #yhat_classes = model.predict_classes(X_test, verbose=0)

    Y_test_bin=[]

    for y_bin in Y_test:
        if (y_bin[0] == 1):
            Y_test_bin.append(0)
        else:
            Y_test_bin.append(1)
    #Y_test_bin = Y_test
            
    yhat_classes = np.argmax(yhat_probs,axis=1)

    print ("Accuracy|Precision|Recall|F1 score|Kappa|ROC AUC|")
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_test_bin, yhat_classes)
    #print('|%f|' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test_bin, yhat_classes)
    #print('%f|' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test_bin, yhat_classes)
    #print('%f|' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test_bin, yhat_classes)
    #print('%f|' % f1)

    # kappa
    kappa = cohen_kappa_score(Y_test_bin, yhat_classes)
    #print('%f|' % kappa)
    # ROC AUC
    auc = roc_auc_score(Y_test, yhat_probs)
    print('%f\t%f\t%f\t%f\t%f\t%f' %(accuracy,precision,recall,f1,kappa,auc))
    # confusion matrix
    matrix = confusion_matrix(Y_test_bin, yhat_classes)
    print(matrix)
    print (Y_test_bin)
    print (yhat_classes)
    return ([accuracy, precision, recall, f1, kappa, auc], matrix)


