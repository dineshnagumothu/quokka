import pandas as pd
import sys
import numpy as np

from classifier_models import model_making, compute_metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

MAX_NB_WORDS = 50000
vocabulary_size=MAX_NB_WORDS
EMBEDDING_DIM = 300

TOPICS_LEN = 8
ENTITIES_LEN = 150
TRIPLES_LEN = 300


import tensorflow as tf

def generate_model(epochs, batch_size,sents=False, topics=False, entities=False, triples=False):
  count=0
  name=""
  if (sents==True):
    count+=1
    name+='Sentences'
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


  if (topics==True and entities==False and triples==False and sents==False):
    train_inputs = train_sampled['topic_probs'].values
    print (train_inputs)
    val_inputs = val['topic_probs'].values
    test_inputs = test['topic_probs'].values

  elif (entities==True and topics==False and triples==False and sents==False):
    train_inputs = train_sampled['entities_text'].values
    train_inputs = tokenizer.texts_to_sequences(train_inputs)
    train_inputs = pad_sequences(train_inputs, maxlen=ENTITIES_LEN)
    print (train_inputs)
    val_inputs = val['entities_text'].values
    val_inputs = tokenizer.texts_to_sequences(val_inputs)
    val_inputs = pad_sequences(val_inputs, maxlen=ENTITIES_LEN)
    test_inputs = test['entities_text'].values
    test_inputs = tokenizer.texts_to_sequences(test_inputs)
    test_inputs = pad_sequences(test_inputs, maxlen=ENTITIES_LEN)

  
  elif (triples==True and topics==False and entities==False and sents==False):
    train_inputs = train_sampled['triple_text'].values
    train_inputs = tokenizer.texts_to_sequences(train_inputs)
    train_inputs = pad_sequences(train_inputs, maxlen=TRIPLES_LEN)
    val_inputs = val['triple_text'].values
    val_inputs = tokenizer.texts_to_sequences(val_inputs)
    val_inputs = pad_sequences(val_inputs, maxlen=TRIPLES_LEN)
    test_inputs = test['triple_text'].values
    test_inputs = tokenizer.texts_to_sequences(test_inputs)
    test_inputs = pad_sequences(test_inputs, maxlen=TRIPLES_LEN)
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
      train_triple_inputs=train_sampled['triple_text'].values
      train_triple_inputs = tokenizer.texts_to_sequences(train_triple_inputs)
      train_triple_inputs = pad_sequences(train_triple_inputs, maxlen=TRIPLES_LEN)
      train_triple_inputs = typeConv(train_triple_inputs)
      train_inputs.append(train_triple_inputs)

      val_triple_inputs=val['triple_text'].values
      val_triple_inputs = tokenizer.texts_to_sequences(val_triple_inputs)
      val_triple_inputs = pad_sequences(val_triple_inputs, maxlen=TRIPLES_LEN)
      val_triple_inputs = typeConv(val_triple_inputs)
      val_inputs.append(val_triple_inputs)

      test_triple_inputs=test['triple_text'].values
      test_triple_inputs = tokenizer.texts_to_sequences(test_triple_inputs)
      test_triple_inputs = pad_sequences(test_triple_inputs, maxlen=TRIPLES_LEN)
      test_triple_inputs = typeConv(test_triple_inputs)
      test_inputs.append(test_triple_inputs)
      
  train_result = tf.keras.utils.to_categorical(train_sampled['relevance'], num_classes=2)
  val_result = tf.keras.utils.to_categorical(val['relevance'], num_classes=2)
  test_result = tf.keras.utils.to_categorical(test['relevance'], num_classes=2)

  model=model_making(count, embedding_matrix, sents=sents,topics=topics, entities=entities, triples=triples)
  
  model.fit(train_inputs,train_sampled['relevance'],epochs=epochs,batch_size=batch_size,validation_data=(val_inputs, val['relevance']))
  
  accr = model.evaluate(test_inputs, test['relevance'])
  metrics, matrix=compute_metrics(model, test_inputs, test_result) 
  print('%f\t%f\t%f\t%f\t%f\t%f' %(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))
  print (matrix)
  return model

def compute_embedding_matrix(tokenizer):
  embeddings_index = dict()
  f = open('GloVe/glove.840B.'+str(EMBEDDING_DIM)+'d.txt', errors='ignore', encoding="utf-8")
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

  if sys.argv[1]=='energyhub':
    filename = 'EH_infersents'
  elif sys.argv[1] == 'reuters':
    filename = 'Reuters_infersents'
  else:
    print ("Wrong dataset name")
    sys.exit()

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
  print (train_sampled)

  model_name = sys.argv[2]

  if (model_name=='sents'):
    model_text = generate_model(epochs=140, batch_size=32,sents=True)
  if (model_name=='topics'):
    model_text = generate_model(epochs=140, batch_size=32,topics=True)
  if (model_name=='entities'):
    model_text = generate_model(epochs=140, batch_size=32,entities=True)
  if (model_name=='triples'):
    model_text = generate_model(epochs=140, batch_size=32,triples=True)
  if (model_name=='sents_triples'):
    model_text = generate_model(epochs=140, batch_size=32,sents=True, triples=True)

