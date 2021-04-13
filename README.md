# quokka

<ol>
<li>

<b> Download data </b> 

Download files from these links and copy them to the data directory

<b>Energy Hub</b>

    Energy Hub Training set - 

    Energy Hub Validation set - 

    Energy Hub Test set - 

<b> Reuters </b>

    Reuters Training set - 

    Reuters Validation set - 

    Retuers Test set - 

</li>
<li>    
<b> Downloading Necessary Packages </b>

  <ul>
    <li> Download NLTK stopwords using

  ```
  import nltk
  
  nltk.download('stopwords')
  ```
  </li>
  <li> Download Mallet from <a href="http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip">here</a>. Unzip and copy it to the directory.

  If you use Google Colab:
  
  ```
  !wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
  !unzip mallet-2.0.8.zip
  ```
  </li>
  
  <li>
    Download GloVe embeddings from <a href="https://nlp.stanford.edu/data/wordvecs/glove.6B.zip">here</a>. Unzip and copy it to the directory.
    
   If you use Google Colab:
    
    ```
    !wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    !unzip glove*.zip
    ```
    
   </li>
  </ul>

  <li>    
<b> Build Topic-Entity Triples </b>

  This step involves
  <ul>
    <li>Training a Topic Modeler over the corpus</li>
    <li>Extracting Named-Entities using spaCy</li>
    <li>Building Triples using Dependency parser and POS tagger</li>
    <li>Apply Topic Entity Filter over these triples</li>
  </ul>
  
  Run the following python file.

  `python data_preprocess.py <dataset>` 
  
  Change `<dataset>` to "energy hub" or "reuters" to select the corpus.
  
  </li>
  
  <li>
  <b> Training Models </b>
  
 Run the following python file.

  `python train.py <dataset> <model>`
  
  Change `<dataset>` to "energy hub" or "reuters" to select the corpus.
  
  Change `<model>` to the following options
  <ul>
    <li>text - for GloVe based text model</li>
    <li>topics - To use topic distributions</li>
    <li>entites - To use Glove-enriched named entities</li>
    <li>triples - To use Glove-enriched triples</li>
    <li>text_topics - To use text and topic distributions</li>
    <li>text_triples - To use text(GloVe) and triples(GloVe)</li>
  </ul>
  
  
  </li>
  </ol>
