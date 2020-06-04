#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


df=pd.read_csv(r'C:\Users\Swarupa V\Desktop\cdsaml_internship\test1.csv')


# In[16]:


df.head()


# In[17]:


df.drop('sl no',axis=1,inplace=True)
df


# In[18]:


X=df.drop('detect',axis=1)
#df.drop('sl no',inplace=True)
y=df['detect']


# In[19]:


tweets=X.copy()
print(tweets)

#tweets.reset_index(inplace=True)
#print(tweets.type())


# In[20]:


import nltk
import re
from nltk.corpus import stopwords


# In[21]:



twets=[]

for i in range(len(tweets)):
    tw=tweets['tweet'][i]
    #print(tw)
    tw=[x for x in tw if x not in stopwords.words('english')]
    tw=''.join(tw)
    #print(tw)
    twets.append(tw)


# In[29]:


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout


# In[30]:


onehot_repr=[one_hot(words,5000)for words in twets] 


# In[31]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='post',maxlen=sent_length)
print(embedded_docs)


# In[38]:


from tensorflow import keras
from tesnorflow.keras import layers


# In[41]:


embedding_vector_features=300
model=keras.Sequential()
model.add(Embedding(5000,embedding_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.8))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())


# In[28]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[184]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=1)


# In[185]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)


# In[186]:


y_pred=model.predict_classes(X_test)


# In[187]:


from sklearn.metrics import confusion_matrix


# In[188]:


confusion_matrix(y_test,y_pred)


# In[189]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[190]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




