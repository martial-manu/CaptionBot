#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.applications.resnet import ResNet50 , preprocess_input , decode_predictions
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model , Model


# In[4]:


model = load_model('./model_weights/model_11.h5')


# In[5]:


model_temp  = ResNet50(weights = "imagenet" , input_shape = (224 , 224 , 3))


# In[6]:


model_resnet = Model(model_temp.input , model_temp.layers[-2].output)


# In[7]:


from tensorflow.keras.preprocessing import image


# In[8]:


def preprocess_img(img):
    img = image.load_img(img , target_size = (224 , 224 , 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img , axis = 0)
    # Normalisation 
    img = preprocess_input(img)  # Resnet model takes images in this format
    return img


# In[9]:


def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape((1 , feature_vector.shape[1] ))
    
    return feature_vector


# In[10]:



# In[11]:



# In[12]:



word_to_idx = joblib.load('./word_to_idx')
idx_to_word = joblib.load('./idx_to_word')


# In[19]:


def predict_caption(photo):
    in_text = "startseq"
    maxlen = 35
    for i in range(maxlen):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence] , maxlen = maxlen , padding = 'post')
        
        ypred = model.predict([photo , sequence])
        ypred = np.argmax(ypred)
        
        word = idx_to_word[ypred]
        in_text = in_text +' '+  word
        
        if word == "endseq":
            break
    final_caption = in_text.split()[1:-1]        
    final_caption = " ".join(final_caption)
    return final_caption


# In[17]:

def caption_the_image(image):
    enc = encode_img(image)
    caption = predict_caption(enc)
    
    return caption



# In[ ]:





# In[20]:





# In[ ]:




