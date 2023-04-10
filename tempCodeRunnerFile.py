import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('housing.csv')

cat_col_list = list(df.select_dtypes('object').columns)
cat_col_list


df_cat = df.select_dtypes('object')
df_cat

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

label_binarizer_cols = []
label_encoder_col = []


