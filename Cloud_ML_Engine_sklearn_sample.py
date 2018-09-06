
# coding: utf-8

# Clud ML Engine に機械学習タスクをローカルから投げるデモ
# 
# 題材として、Kaggle の titanic データ（ https://www.kaggle.com/c/titanic/data ）の生存者分類機をランダムフォレストで作成して予測してみます。
# 
# 手順としては、
# 
# 1. データの前処理をして Cloud Storage にデータを保存
# 2. 学習用コードを書いて ML Engine にジョブを投げる
# 3. Cloud Storage 上に作成されたモデルをダウンロードしてローカルでモデルをロードして予測
# 
# という流れで進めています。

# ## 1. 準備

# ## 1.1 Cloud Storage にバケットを作成する
# 
# データセットを格納するバケットの作成。  
# バケット名は何でもよいが、同じバケット名は使用できないので注意

# In[1]:

# 何故か環境変数のPATHが通ってないので google api alient の json ファイルのパスを通しておく。  
# この辺は個々人の環境設定によるかもしれません。

import os 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'YOUR/PATH/SERVICE_ACCOUNT_KEY.json'


# In[2]:

# 後述で subprocess を使って実行している箇所もあります。
# せっかくPythonライブラリもあるのでこっちも使ってみたかった。
from google.cloud import storage as gcs
from googleapiclient import errors

project_name = 'YOUR_PROJECT_NAME'
gcs_client = gcs.Client(project_name)

bucket_name = 'sklearn-sample'

# バケット作成
# すでに存在するバケット名の場合は作成エラーになるので回避
if bucket_name not in [b.name for b in gcs_client.list_buckets()]:
    bucket = gcs_client.create_bucket(bucket_name)
    print('Bucket {} created.'.format(bucket.name))
else:
    print("You already own bucket {}.".format(bucket_name))


# ## 1.2 データセットをダウンロードしてCloud Storageに格納する

# kaggle サイトからデータセットをダウンロード

# In[3]:

get_ipython().run_cell_magic('bash', '', 'DATA_DIR="${PWD}/dataset/"\nkaggle competitions download -c titanic -p $DATA_DIR --force ')


# In[4]:

import numpy as np
import pandas as pd 

# アップロード用関数を作成する。サブディレクトリを指定して送信できるようひと工夫。
def file_upload(local_filepath, bucket, bucket_filename=None):
    """
    bucket: 格納先バケットオブジェクト
    bucket_filename: バケット先に設定したいパスを含んだファイル名
    """
    if bucket_filename is None:
        # 名前がないときはtmpディレクトリを作成してローカルのファイル名と同じで格納する
        bucket_filename = 'tmp/' + os.path.basename(local_filepath)
    blob = bucket.blob(bucket_filename)
    blob.upload_from_filename(local_filepath)
    
    
# データ前処理を行う関数
def preprocessing(df):
    #欠損値処理
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    #カテゴリ変数の変換
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    df = df.drop(['Cabin','Name','PassengerId','Ticket','Embarked'],axis=1)
    df = df.astype('float64')
    
    return df 


# In[5]:

# データ前処理    
train_data = './dataset/train.csv'
test_data = './dataset/test.csv'

df_train = pd.read_csv(train_data)
df_test = pd.read_csv(test_data)

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

# 前処理結果を上書きで保存
df_train.to_csv(train_data, index=False)
df_train.to_csv(test_data, index=False)


# データ送信
bucket = gcs_client.bucket(bucket_name)

file_upload(train_data, bucket, bucket_filename='dataset/train.csv')
file_upload(test_data, bucket, bucket_filename='dataset/test.csv') 


# # 2. モデルとトレーニングの記述

# ## 2.1 初期設定
# 
# jupyter に書いた任意のセルをコードにするマジックを使う。  
# `./lib/mlcodemagic.py` にあるのでそれを使う。  
# （自作したクラスなので pip install とかできません）  
# 
# 
# - `%%mlcodes` でコードを記述。`run_local` と続けて記載するとローカルのjupyterでも同じコードを動かす。  
#   - 重たい学習で、ローカルで動かしたくないときは run_local を書かない
# - `%code_to_pyfile` でコードを `./trainer/task.py` に保存。これを行うまではコードは保存されない。  
# - `%clear_mlcode` でコードの初期化。`%%mlcodes` のセルを叩くたびに append されるので、同じセルをたくさん叩いちゃったときに直す用。

# In[6]:

# マジックを有効にする
from lib.mlcodemagic import MLCodeMagic 
ip = get_ipython()
ip.register_magics(MLCodeMagic)


# ## 2.2 モデル記述

# In[7]:

get_ipython().run_cell_magic('mlcodes', '', '\nimport numpy as np\nimport pandas as pd \nfrom sklearn.model_selection import train_test_split\nfrom tensorflow.python.lib.io import file_io\n\nproject_name = \'YOUR_PROJECT_NAME\'\nbucket_path = "gs://sklearn-sample"\ntrain_file_path = "{}/dataset/train.csv".format(bucket_path)\n\n# get train data \n# CloudML上からgoogle cloud storage上のファイルにアクセスする際には、tensorflowのfile_ioパッケージを使う必要がある\nwith file_io.FileIO(train_file_path,\'r\') as f: \n    df = pd.read_csv(f)\n\n# train valid split \ny = np.array(df[\'Survived\'], dtype=\'int8\')\ndel df[\'Survived\']\nX = np.array(df, dtype=\'float64\')\n\nX_train, X_validation, y_train, y_validation = train_test_split(\n    X, y, test_size=0.2, random_state=0)')


# In[8]:

# ローカル実行用データ用意
# デモで run_local 表示するために今回は用意している。実際にはなくてもいい。

from sklearn.model_selection import train_test_split

y = np.array(df_train['Survived'], dtype='int8')
del df_train['Survived']
X = np.array(df_train, dtype='float64')

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=0)


# In[9]:

get_ipython().run_cell_magic('mlcodes', 'run_local', '\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\n\n# make model\nclf = RandomForestClassifier()\nclf.fit(X_train, y_train)\n\ny_pred = clf.predict(X_validation)\nacc = accuracy_score(y_validation, y_pred)\n\nprint("accuracy: {:.4f}".format(acc))')


# In[10]:

get_ipython().run_cell_magic('mlcodes', '', "\nimport os \nimport pickle \n\n# save model \nmodel_name = 'model.pkl'\n\nwith file_io.FileIO(os.path.join(bucket_path,'model',model_name), 'wb') as model_file:\n    pickle.dump(clf, model_file)")


# モデルコードをpyファイルに変換する

# In[11]:

get_ipython().magic('code_to_pyfile')


# ## 2.3 学習実行

# In[13]:

import os 
from datetime import datetime 
import subprocess

# ジョブ定義
# ジョブの名前は何でもいいです。 今回は job_[日付] にしました。
bucket_path = "gs://sklearn-sample"
job_name = datetime.now().strftime("job_%Y%m%d_%H%M%S")
job_dir = os.path.join(bucket_path, job_name)

# runtime-version 1.4 以上でないと python 3.5 が使えないので注意
# scale-tier BASIC_GPU で GPU指定可能
job_cmd = """gcloud ml-engine jobs submit training {0}
--job-dir {1}
--module-name trainer.task
--package-path trainer 
--staging-bucket {2}
--region us-central1
--runtime-version 1.6
--python-version 3.5
--scale-tier BASIC_GPU
""".format(job_name, job_dir, bucket_path)


# In[ ]:

# ジョブ実行
subprocess.run(job_cmd.split())


# ここまで実行したらジョブが終了するまで待っていてください。

# # 3. 学習済みモデルのダウンロード

# In[ ]:

# 学習済みモデルをローカルにダウンロード
model_dir = "./models"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_name = 'model.pkl'
gcs_model_path = os.path.join(bucket_path, 'model', model_name)
    
dl_cmd = "gsutil cp {0} {1}".format(gcs_model_path, model_dir)
subprocess.run(dl_cmd.split())


# In[ ]:

import pickle 

# モデルのロード
with open(os.path.join(model_dir, model_name),'rb') as f:
    clf = pickle.load(f)


# In[25]:

# 予測する
X_test = np.array(df_test, dtype='float64')

y_pred = clf.predict(X_test)

print(y_pred[0:10])


# In[ ]:



