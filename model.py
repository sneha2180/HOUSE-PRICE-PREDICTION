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

for each_col in df_cat.columns:
    if len(df_cat[each_col].unique()) == 2:
        label_binarizer_cols.append(each_col)
    else:
        label_encoder_col.append(each_col)
    print(df_cat[each_col].unique())

lb_obj = {}
for each_col in label_binarizer_cols:
    lb = LabelBinarizer()
    lb_obj[each_col] = lb
    df[each_col] = lb.fit_transform(df[each_col])

le_obj = {}
for each_col in label_encoder_col:
    le = LabelEncoder()
    le_obj[each_col] = le
    df[each_col] = le.fit_transform(df[each_col])

x = df.drop('price', axis = 1)
y = df['price']

# Model Building using 3 different algorithms

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
# Linear Regression model
# Assigning the algorithm to the variable
lr = LinearRegression()

# Fitting of the model
lr.fit(x,y)
lr.predict(x)[1:50]