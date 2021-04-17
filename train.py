import pandas as pd
import sys
import numpy as np

from classifier_models import model_making, compute_metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

import argparse


MAX_NB_WORDS = 50000
vocabulary_size=MAX_NB_WORDS
EMBEDDING_DIM = 300

TOPICS_LEN = 100
TEXT_LEN = 1000
ENTITIES_LEN = 1000
TRIPLES_LEN = 1000

triple_col = 'openie_triple_text'

import tensorflow as tf

def generate_model(epochs, batch_size,sents=False, topics=False, entities=False, triples=False, text=False):
  count=0
  name=""
  if (sents==True):
    count+=1
    name+='Sentences'
  if (text==True):
    count+=1
    name+='Text'
  if (topics==True):
    count+=1
    if(count>1):
      name+='_'
    name+='Topics'
  if (entities==True):
    count+=1
    if(count>1):
      name+='_'
    name+='Entities'
  if (triples==True):
    count+=1
    if(count>1):
      name+='_'
    name+='Triples'
  
  #name+='_'+str(i)

  print (name)
  
  #logdir = os.path.join("EH_logs", name)
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

  embedding_matrix= compute_embedding_matrix(tokenizer)

  ##if only text is selected
  if (topics==False and entities==False and triples==False and text==True):
    train_inputs = train_sampled['text'].values
    train_inputs = tokenizer.texts_to_sequences(train_inputs)
    train_inputs = pad_sequences(train_inputs, maxlen=TEXT_LEN)
    val_inputs = val['text'].values
    val_inputs = tokenizer.texts_to_sequences(val_inputs)
    val_inputs = pad_sequences(val_inputs, maxlen=TEXT_LEN)
    test_inputs = test['text'].values
    test_inputs = tokenizer.texts_to_sequences(test_inputs)
    test_inputs = pad_sequences(test_inputs, maxlen=TEXT_LEN)

  ##if only topics are selected
  if (topics==True and entities==False and triples==False and sents==False):
    train_inputs = train_sampled['topic_probs'].values
    val_inputs = val['topic_probs'].values
    test_inputs = test['topic_probs'].values

  ##if only entities are selected
  elif (entities==True and topics==False and triples==False and sents==False):
    train_inputs = train_sampled['entities_text'].values
    train_inputs = tokenizer.texts_to_sequences(train_inputs)
    train_inputs = pad_sequences(train_inputs, maxlen=ENTITIES_LEN)
    val_inputs = val['entities_text'].values
    val_inputs = tokenizer.texts_to_sequences(val_inputs)
    val_inputs = pad_sequences(val_inputs, maxlen=ENTITIES_LEN)
    test_inputs = test['entities_text'].values
    test_inputs = tokenizer.texts_to_sequences(test_inputs)
    test_inputs = pad_sequences(test_inputs, maxlen=ENTITIES_LEN)

  ##if only triples are selected
  elif (triples==True and topics==False and entities==False and sents==False):
    train_inputs = train_sampled[triple_col].values
    train_inputs = tokenizer.texts_to_sequences(train_inputs)
    train_inputs = pad_sequences(train_inputs, maxlen=TRIPLES_LEN)
    val_inputs = val[triple_col].values
    val_inputs = tokenizer.texts_to_sequences(val_inputs)
    val_inputs = pad_sequences(val_inputs, maxlen=TRIPLES_LEN)
    test_inputs = test[triple_col].values
    test_inputs = tokenizer.texts_to_sequences(test_inputs)
    test_inputs = pad_sequences(test_inputs, maxlen=TRIPLES_LEN)
  
  ##if only sentence embeddings are selected
  elif (triples==False and topics==False and entities==False and sents==True):
    train_inputs = train_sampled['sent_embeddings'].values
    val_inputs=val['sent_embeddings']
    test_inputs=test['sent_embeddings']
  if(count==1):
    train_inputs = typeConv(train_inputs)
    val_inputs = typeConv(val_inputs)
    test_inputs = typeConv(test_inputs)

  if(count>1):
    train_inputs = []
    val_inputs = []
    test_inputs = []

    if (text):
      train_text_inputs=train_sampled['text'].values
      train_text_inputs = tokenizer.texts_to_sequences(train_text_inputs)
      train_text_inputs = pad_sequences(train_text_inputs, maxlen=TEXT_LEN)
      train_text_inputs = typeConv(train_text_inputs)
      train_inputs.append(train_text_inputs)

      val_text_inputs=val['text'].values
      val_text_inputs = tokenizer.texts_to_sequences(val_text_inputs)
      val_text_inputs = pad_sequences(val_text_inputs, maxlen=TEXT_LEN)
      val_text_inputs = typeConv(val_text_inputs)
      val_inputs.append(val_text_inputs)

      test_text_inputs=test['text'].values
      test_text_inputs = tokenizer.texts_to_sequences(test_text_inputs)
      test_text_inputs = pad_sequences(test_text_inputs, maxlen=TEXT_LEN)
      test_text_inputs = typeConv(test_text_inputs)
      test_inputs.append(test_text_inputs)

    if (sents):
      train_sents_inputs=train_sampled['sent_embeddings'].values
      train_sents_inputs = typeConv(train_sents_inputs)
      train_inputs.append(train_sents_inputs)
      val_sents_inputs=val['sent_embeddings'].values
      val_sents_inputs = typeConv(val_sents_inputs)
      val_inputs.append(val_sents_inputs)
      test_sents_inputs=test['sent_embeddings'].values
      test_sents_inputs = typeConv(test_sents_inputs)
      test_inputs.append(test_sents_inputs)

    if (topics):
      train_topic_inputs=train_sampled['topic_probs'].values
      train_topic_inputs = typeConv(train_topic_inputs)
      train_inputs.append(train_topic_inputs)

      val_topic_inputs=val['topic_probs'].values
      val_topic_inputs = typeConv(val_topic_inputs)
      val_inputs.append(val_topic_inputs)

      test_topic_inputs=test['topic_probs'].values
      test_topic_inputs = typeConv(test_topic_inputs)
      test_inputs.append(test_topic_inputs)
    if (entities):
      train_entity_inputs=train_sampled['entities_text'].values
      train_entity_inputs = tokenizer.texts_to_sequences(train_entity_inputs)
      train_entity_inputs = pad_sequences(train_entity_inputs, maxlen=ENTITIES_LEN)
      train_entity_inputs = typeConv(train_entity_inputs)
      train_inputs.append(train_entity_inputs)

      val_entity_inputs=val['entities_text'].values
      val_entity_inputs = tokenizer.texts_to_sequences(val_entity_inputs)
      val_entity_inputs = pad_sequences(val_entity_inputs, maxlen=ENTITIES_LEN)
      val_entity_inputs = typeConv(val_entity_inputs)
      val_inputs.append(val_entity_inputs)

      test_entity_inputs=test['entities_text'].values
      test_entity_inputs = tokenizer.texts_to_sequences(test_entity_inputs)
      test_entity_inputs = pad_sequences(test_entity_inputs, maxlen=ENTITIES_LEN)
      test_entity_inputs = typeConv(test_entity_inputs)
      test_inputs.append(test_entity_inputs)

    if (triples):
      train_triple_inputs=train_sampled[triple_col].values
      train_triple_inputs = tokenizer.texts_to_sequences(train_triple_inputs)
      train_triple_inputs = pad_sequences(train_triple_inputs, maxlen=TRIPLES_LEN)
      train_triple_inputs = typeConv(train_triple_inputs)
      train_inputs.append(train_triple_inputs)

      val_triple_inputs=val[triple_col].values
      val_triple_inputs = tokenizer.texts_to_sequences(val_triple_inputs)
      val_triple_inputs = pad_sequences(val_triple_inputs, maxlen=TRIPLES_LEN)
      val_triple_inputs = typeConv(val_triple_inputs)
      val_inputs.append(val_triple_inputs)

      test_triple_inputs=test[triple_col].values
      test_triple_inputs = tokenizer.texts_to_sequences(test_triple_inputs)
      test_triple_inputs = pad_sequences(test_triple_inputs, maxlen=TRIPLES_LEN)
      test_triple_inputs = typeConv(test_triple_inputs)
      test_inputs.append(test_triple_inputs)
      
  train_result = tf.keras.utils.to_categorical(train_sampled['relevance'], num_classes=2)
  val_result = tf.keras.utils.to_categorical(val['relevance'], num_classes=2)
  test_result = tf.keras.utils.to_categorical(test['relevance'], num_classes=2)

  model=model_making(count, embedding_matrix, sents=sents,topics=topics, entities=entities, triples=triples, text=text, fine_tune=False)
  
  es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=5, restore_best_weights=True)
  model.fit(train_inputs,train_sampled['relevance'],epochs=epochs,batch_size=batch_size,validation_data=(val_inputs, val['relevance']), callbacks=[es])
  tf.keras.utils.plot_model(model, to_file='model_plots/'+name+'.png', show_shapes=True, show_layer_names=True)
  
  accr = model.evaluate(test_inputs, test['relevance'])
  metrics, matrix=compute_metrics(model, test_inputs, test_result) 
  print('%f\t%f\t%f\t%f\t%f\t%f' %(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))
  print (matrix)
  return model

def compute_embedding_matrix(tokenizer):
  embeddings_index = dict()
  f = open('glove.6B.'+str(EMBEDDING_DIM)+'d.txt', errors='ignore', encoding="utf-8")
  for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  f.close()

  embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
  for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
      break
    else:
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[index] = embedding_vector
  return embedding_matrix

def typeConv(data):
  X_train_array = []
  for x in data:
    x = np.asarray(x).astype(np.float32)
    X_train_array.append(x)

  X_train_array=np.asarray(X_train_array)
  return X_train_array

if __name__ == "__main__":

  parser = argparse.ArgumentParser()   
  parser.add_argument('--dataset', required=True, help="Dataset required")
  parser.add_argument('--model', required=True, help="Choose a model name")
  
  args = parser.parse_args()

  print(f'{args.dataset} selected')

  if args.dataset=='energyhub':
    filename = 'EH_infersents'
  elif args.dataset == 'reuters':
    filename = 'Reuters_infersents'
  else:
    print ("Wrong dataset name")
    sys.exit()

  model_name = args.model

  print ("Reading Data")

  train= pd.read_json(r"data/"+filename+"_train_probs.json")
  val= pd.read_json(r"data/"+filename+"_val_probs.json")
  test= pd.read_json(r"data/"+filename+"_test_probs.json")

  tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~', lower=True)


  token_text=train['triple_text'].values.tolist()
  token_text.extend(val['triple_text'].values)
  token_text.extend(test['triple_text'].values.tolist())

  token_text.extend(train['topics_text'].values)
  token_text.extend(val['topics_text'].values)
  token_text.extend(test['topics_text'].values)

  token_text.extend(train['entities_text'].values)
  token_text.extend(val['entities_text'].values)
  token_text.extend(test['entities_text'].values)

  tokenizer.fit_on_texts(token_text)

  word_index = tokenizer.word_index

  rel_count = train.relevance.value_counts()

  print (rel_count)

  sample_size=0
  if(rel_count[0]>rel_count[1]):
    sample_size=rel_count[1]
  else:
    sample_size=rel_count[0]

  nd_majority = train[train.relevance==1]
  nd_minority = train[train.relevance==0]
  # Downsample majority class
  nd_majority_downsampled = resample(nd_majority, 
                                  replace=False,    # sample without replacement
                                  n_samples=sample_size,     # to match minority class
                                  random_state=2020) # reproducible results

  train_sampled = pd.concat([nd_majority_downsampled, nd_minority])
  #print (train_sampled)

  if (model_name=='sents'):
    model_text = generate_model(epochs=140, batch_size=32,sents=True)
  elif (model_name=='text'):
    model_text = generate_model(epochs=140, batch_size=32,text=True)
  elif (model_name=='topics'):
    model_text = generate_model(epochs=140, batch_size=32,topics=True)
  elif (model_name=='entities'):
    model_text = generate_model(epochs=140, batch_size=32,entities=True)
  elif (model_name=='triples'):
    model_text = generate_model(epochs=140, batch_size=32,triples=True)
  elif (model_name=='text_triples'):
    model_text = generate_model(epochs=140, batch_size=32,text=True, triples=True)
  elif (model_name=='text_topics'):
    model_text = generate_model(epochs=140, batch_size=32,text=True, topics=True)
  elif (model_name=='text_entities'):
    model_text = generate_model(epochs=140, batch_size=32,text=True, entities=True)
  elif (model_name=='text_topics_entities'):
    model_text = generate_model(epochs=140, batch_size=32,text=True, topics=True, entities=True)
  elif (model_name=='text_entities_triples'):
    model_text = generate_model(epochs=140, batch_size=32,text=True, topics=True, entities=True)
  elif (model_name=='sents_triples'):
    model_text = generate_model(epochs=140, batch_size=32,sents=True, triples=True)
  else:
    print ("Wrong model selected")

