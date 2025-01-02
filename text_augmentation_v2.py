import os
os.environ["MODEL_DIR"] = 'embeddings/'

# from nlpaug.util.file.download import DownloadUtil
# DownloadUtil.download_word2vec(dest_dir='embeddings') # Download word2vec model
# DownloadUtil.download_glove(model_name='glove.6B', dest_dir='embeddings') # Download GloVe model
# DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='embeddings') # Download fasttext model

import time 
import pandas as pd
import numpy as np
import itertools
# from sklearn.model_selection import train_test_split

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action
from skmultilearn.model_selection import iterative_train_test_split

character_augmenter_list = ['ocr','keyboard','random_insert','random_substitute',
    'random_swap','random_delete']
word_augmenter_list = ['spelling','word2vec','glove','fasttext',
    'bert','distilbert','roberta','wordnet']
sentence_augmenter_list = ['xlnet','gpt2','distilgpt2']


def character_augmenter(text,method='ocr'):
    augmenters = {
        'ocr':nac.OcrAug(),
        'keyboard':nac.KeyboardAug(),
        'random_insert':nac.RandomCharAug(action="insert"),
        'random_substitute':nac.RandomCharAug(action="substitute"),
        'random_swap':nac.RandomCharAug(action="swap"),
        'random_delete':nac.RandomCharAug(action="delete")
    }
    augmented_texts = augmenters[method].augment(text)
    return augmented_texts

def word_augmenter(text,method='word2vec'):
    augmenters = {
        'spelling':naw.SpellingAug(),
        'word2vec':naw.WordEmbsAug(
            model_type='fasttext', 
            model_path=os.environ.get("MODEL_DIR")+'crawl-300d-2M.vec',
            action="substitute"),
        'glove':naw.WordEmbsAug(
            model_type='glove', 
            model_path=os.environ.get("MODEL_DIR")+'glove.6B.300d.txt',
            action="substitute"),
        'fasttext':naw.WordEmbsAug(
            model_type='fasttext', 
            model_path=os.environ.get("MODEL_DIR")+'wiki-news-300d-1M.vec',
            action="substitute"),
        # 'tfidf':naw.TfIdfAug(
        #     model_path=os.environ.get("MODEL_DIR"),
        #     action="substitute"),
        'bert':naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="substitute",device='cuda'),
        'distilbert':naw.ContextualWordEmbsAug(
            model_path='distilbert-base-uncased', action="substitute",device='cuda'),
        'roberta':naw.ContextualWordEmbsAug(
            model_path='roberta-base', action="substitute",device='cuda'),
        'wordnet':naw.SynonymAug(aug_src='wordnet')
    }
    # augmented_texts = augmenters[method].augment(text)
    if len(text)>100:
      data = text
      chunks = [data[x:x+50] for x in range(0, len(data), 50)]
      augmented_texts = []
      current = 0
      total = len(chunks)
      for text in chunks:
        augmented_text = augmenters[method].augment(text)
        augmented_texts+=augmented_text
        current += 1
        progressBar(current, total, barLength = 20)
    else:
      augmented_texts = augmenters[method].augment(text)
    return augmented_texts

def sentence_augmenter(text,method='gpt2'):
    augmenters = {
        'xlnet':nas.ContextualWordEmbsForSentenceAug(model_path='xlnet-base-cased',device='cuda',
            force_reload=True),
        'gpt2':nas.ContextualWordEmbsForSentenceAug(model_path='gpt2'),
        'distilgpt2':nas.ContextualWordEmbsForSentenceAug(model_path='distilgpt2')
    }
    # augmented_texts = augmenters[method].augment(text)
    if len(text)>100:
      data = text
      chunks = [data[x:x+50] for x in range(0, len(data), 50)]
      augmented_texts = []
      current = 0
      total = len(chunks)
      for text in chunks:
        augmented_text = augmenters[method].augment(text)
        augmented_texts+=augmented_text
        current += 1
        progressBar(current, total, barLength = 20)
    else:
      augmented_texts = augmenters[method].augment(text)
    return augmented_texts

def exam_augmentation(methods,input_path='data.csv',output_path='augmented_data/'):
    # categories = ['anger','disgust','fear','joy','sadness']
    df = pd.read_csv(input_path) 
    labels = [label for label in df.keys() if label not in ['id', 'text']]
    # id2label = {idx:label for idx, label in enumerate(labels)}
    # label2id = {label:idx for idx, label in enumerate(labels)}
       

    if len(labels)!=0 and len(labels)!=9:
        # update the df with selected labels
        df_emotion = df[[column for column in df.columns.tolist() if column in (labels)]]
        # print(df_emotion.value_counts())
    # print(df.text.tolist())
    X_train, y_train, X_test, y_test = iterative_train_test_split(np.array(df.text.tolist()).reshape(-1,1), 
        np.array(df_emotion.values.tolist()) , test_size = 0.5)
    # X_train = df.text.tolist()
    # y_train = df_emotion.values.tolist() 
    X_train = X_train.reshape(-1).tolist()
    categories = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
    results=[]
    for text, predictions in zip(X_train,y_train.tolist()):
        result = dict(zip(categories, predictions))
        result['text'] =  text
        results.append(result)
    df = pd.DataFrame(results)
    df = df[['text','Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]
    df.to_csv(output_path+f'X_train.csv',index=False)
    # print(X_train)
    for method in methods:
        print(method)
        df_new = get_aug_and_label(X_train,y_train,method)
        # print(df_new.head())
        df_new.to_csv(output_path+f'{method}_train.csv',index=False)
    pass

def get_aug_and_label(texts,labels,method='ocr'):
    if method in character_augmenter_list:
        augmented_texts = character_augmenter(texts,method)
    elif method in word_augmenter_list:
        augmented_texts = word_augmenter(texts,method)
    elif method in sentence_augmenter_list:
        augmented_texts = sentence_augmenter(texts,method)
    else:
        print('method does not exist')
    # new_texts = texts+augmented_texts
    # new_labels = np.concatenate((np.array(labels),np.array(labels)), axis=0).tolist()
    new_texts = augmented_texts
    new_labels = labels
    # print(len(labels),len(new_labels))
    results = []
    categories = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
    for text, predictions in zip(new_texts,new_labels):
        result = dict(zip(categories, predictions))
        result['text'] =  text
        results.append(result)
    df = pd.DataFrame(results)
    df = df[['text','Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']]
    return df

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %f %%' % (arrow, spaces, percent), end='\r')
    pass

if __name__ == '__main__':
    # exam_augmentation(character_augmenter_list,input_path='semeval2025-task11/train/track_a/eng.csv',output_path='semeval2025-task11/new_train/')
    exam_augmentation(word_augmenter_list,input_path='semeval2025-task11/train/track_a/eng.csv',output_path='semeval2025-task11/new_train/')
    exam_augmentation(sentence_augmenter_list,input_path='semeval2025-task11/train/track_a/eng.csv',output_path='semeval2025-task11/new_train/')
    # for lst in [character_augmenter_list,word_augmenter_list,sentence_augmenter_list]:
    #     exam_augmentation(lst,input_path='semeval2025-task11/train/track_a/eng.csv',output_path='semeval2025-task11/new_train/')

