from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import pandas as pd

def iter_sat_split(input_data='data.csv',output_path='splited_data/'):
    # categories = ['anger','disgust','fear','joy','sadness']
    df = pd.read_csv(input_data) 
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


