#!/usr/bin/env python
# coding: utf-8

# In[82]:


import json
import math
import numpy as np
import random
import sklearn
import string
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn import metrics
#from gensim.models import Word2Vec
import dateutil
from scipy.sparse import lil_matrix # To build sparse feature matrices, if you like
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


dataset = json.load(open("train-transcripts-aligned.json", 'r+'))


# In[109]:


datasetTest = json.load(open("test-transcripts-aligned.json", 'r+'))


# In[121]:


test_durations = []
test_utterance_lengths = []
test_offset = []
test_has_q = []
test_has_questions = []  # List to store 'has_q' feature
test_num_words = []
test_num_sentences = []
test_speaker_one_hot = []
for episode_id, turns in datasetTest.items():
    for turn in turns:
        duration = turn['utterance_end'] - turn['utterance_start']
        utterance_length = len(turn['utterance'])
        has_question = int(turn['has_q'])
        words = turn['n_words']
        sentences = turn['n_sentences']
        
        if turn['speaker'] in top100.keys():
            oneHot = top100[turn['speaker']]
        else:
            oneHot = top100['other']
        
        test_offset.append(1)
        
        test_durations.append(duration)
        test_utterance_lengths.append(utterance_length)
        test_has_questions.append(has_question)
        test_num_words.append(words)
        test_num_sentences.append(sentences)
        test_speaker_one_hot.append(oneHot)
        
y_test = np.array(test_durations)


# In[4]:


print(dataset['ep-1'][0]['duration'])
print(dataset['ep-1'][0])


# In[5]:


actNames = defaultdict(int)
actTitles = defaultdict(int)
roles = defaultdict(int)
speakersUtterances = defaultdict(int)
episodeEntries = defaultdict(int)
totalUtterances = 0
for x in dataset:
    for y in dataset[x]:
        actNames[y['act']] += 1
        actTitles[y['act_title']] += 1
        roles[y['role']] += 1
        speakersUtterances[y['speaker']] += 1
        episodeEntries[y['episode']] += 1       
        totalUtterances += 1


# In[6]:


print(totalUtterances)


# In[7]:


categories = list(actNames.keys())
values = [entry for entry in actNames.values()]
colors = [('#%06x' % random.randint(0, 0xFFFFFF)) for _ in range(len(categories))]

# Create a bar graph
plt.bar(categories, values, color=colors)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Randomized Colors')


# In[8]:


categories = list(roles.keys())
values = [entry for entry in roles.values()]
colors = [('#%06x' % random.randint(0, 0xFFFFFF)) for _ in range(len(categories))]

# Create a bar graph
plt.bar(categories, values, color=colors)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Randomized Colors')


# In[9]:


interviewer_length = []
host_length = []
subject_length = []
total_length = []
data = dataset['ep-1']

for x in dataset:
    for y in dataset[x]:
        if y['role'] == 'interviewer':
            interviewer_length.append(y['duration'])
        elif y['role'] == 'host':
            host_length.append(y['duration'])
        else:
            subject_length.append(y['duration'])
        total_length.append(y['duration'])


# In[10]:


i_length = sum(interviewer_length)
h_length = sum(host_length)
s_length = sum(subject_length)
t_length = sum(total_length)
print(i_length, h_length, s_length, t_length)
print("percent of interviewar utterances: ", i_length/t_length)
print("percent of host utterances: ", h_length/t_length)
print("percent of subject utterances: ", s_length/t_length)

i_average = i_length/len(interviewer_length)
h_average = h_length/len(host_length)
s_average = s_length/len(subject_length)
t_average = t_length/len(total_length)
print('average length of utterances interviewer, host, subject, total: ', i_average, h_average, s_average, t_average)


# In[11]:


sorted_data = dict(sorted(speakersUtterances.items(), key=lambda x: x[1], reverse=True)[:10])

categories = list(sorted_data.keys())
values = list(sorted_data.values())
colors = [('#%06x' % random.randint(0, 0xFFFFFF)) for _ in range(len(categories))]

# Create a bar graph
plt.bar(categories, values, color=colors)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Randomized Colors')


# In[12]:


sorted_dataSpeaker = dict(sorted(speakersUtterances.items(), key=lambda x: x[1], reverse=True)[:100])
for key, value in list(sorted_dataSpeaker.items())[:100]:
    print(f'{key}: {value}')


# In[13]:


speakerTime = defaultdict(int)

for x in dataset:
    for y in dataset[x]:
        speakerTime[y['speaker']] += y['duration']


# In[14]:


sorted_data = dict(sorted(speakerTime.items(), key=lambda x: x[1], reverse=True)[:10])

categories = list(sorted_data.keys())
values = list(sorted_data.values())
colors = [('#%06x' % random.randint(0, 0xFFFFFF)) for _ in range(len(categories))]

# Create a bar graph
plt.bar(categories, values, color=colors)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Randomized Colors')


# In[15]:


sorted_dataSpeaker = dict(sorted(speakerTime.items(), key=lambda x: x[1], reverse=True)[:100])
for key, value in list(sorted_dataSpeaker.items())[:100]:
    print(f'{key}: {value}')


# In[16]:


question_length = []
total_length = []
statement_length = []
data = dataset['ep-1']

for d in data:
    if d['has_q'] is True:
        question_length.append(d['duration'])
    else:
        statement_length.append(d['duration'])
    total_length.append(d['duration'])


# In[17]:


# question utterance duration is about 0.775 times shorter than statement utterance duration
q_length = sum(question_length)
t_length = sum(total_length)
s_length = sum(statement_length)
print(q_length, s_length, t_length)
print(s_length/q_length)
print(t_length/q_length)

q_average = q_length/len(question_length)
s_average = s_length/len(statement_length)
t_average = t_length/len(total_length)
print('average q,s,t: ', q_average, s_average, t_average)
print(q_average/s_average)


# ## Learning Utterance Duration

# In[18]:


def MSE(ypred,y):
    diffs = [(a-b)**2 for (a,b) in zip(y,ypred)]
    return sum(diffs) / len(diffs)


# In[19]:


def MAE(ypred,y):
    diffs = [abs(a-b) for (a,b) in zip(y,ypred)]
    return sum(diffs)/len(diffs)


# In[111]:


offset = []
utterance_lengths = []
durations = []

# Iterating through each episode
for episode_id, turns in dataset.items():
    # Iterating through each turn in the episode
    for turn in turns:
        utterance_length = len(turn['utterance'])
        duration = turn['utterance_end'] - turn['utterance_start']

        offset.append(1)
        utterance_lengths.append(utterance_length)
        durations.append(duration)
        
X_train = np.array(utterance_lengths).reshape(-1, 1)  # Features (utterance lengths)
y_train = np.array(durations)  # Target (durations)

print("X sample:", X[:5])  # Print first 5 samples of features
print("y sample:", y[:5])  # Print first 5 samples of targets


# In[119]:


# Split the data into training and testing sets
X_test = np.array(test_utterance_lengths).reshape(-1, 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model (optional)
y_pred = model.predict(X_test)
lengthOnly = MAE(y_pred,y_test)


# In[120]:


print("MAE with only length of utterance as a feature: ", lengthOnly)


# In[122]:


offset = []
utterance_lengths = []
durations = []
has_questions = []  # List to store 'has_q' feature

# Iterating through each episode
for episode_id, turns in dataset.items():
    # Iterating through each turn in the episode
    for turn in turns:
        utterance_length = len(turn['utterance'])
        duration = turn['utterance_end'] - turn['utterance_start']
        has_question = int(turn['has_q'])  # Convert boolean to int (True to 1, False to 0)

        offset.append(1)
        utterance_lengths.append(utterance_length)
        durations.append(duration)
        has_questions.append(has_question)
        
X2_train = np.column_stack((offset, utterance_lengths, has_questions))  # Combine utterance lengths and has_q features
y2_train = np.array(durations)  # Target (durations)


# In[123]:


# Split the data into training and testing sets
X2_test = np.column_stack((test_offset, test_utterance_lengths, test_has_questions))
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X2_train, y2_train)

# Test the model (optional)
y2_pred = model.predict(X2_test)
lengthAndQuestion = MAE(y2_pred,y_test)


# In[124]:


print("MAE including the feature of if the utterance has a question: ", lengthAndQuestion)


# In[127]:


offset = []
utterance_lengths = []
durations = []
has_questions = []  # List to store 'has_q' feature
num_words = []

# Iterating through each episode
for episode_id, turns in dataset.items():
    # Iterating through each turn in the episode
    for turn in turns:
        utterance_length = len(turn['utterance'])
        duration = turn['duration']
        has_question = int(turn['has_q'])  # Convert boolean to int (True to 1, False to 0)
        words = turn['n_words']

        offset.append(1)
        utterance_lengths.append(utterance_length)
        durations.append(duration)
        has_questions.append(has_question)
        num_words.append(words)
        
X3_train = np.column_stack((offset, utterance_lengths, has_questions, num_words))  # Combine utterance lengths and has_q features
y3_train = np.array(durations)  # Target (durations)


# In[128]:


X3_test = np.column_stack((test_offset, test_utterance_lengths, test_has_questions, test_num_words))

# Create and train the model
model = LinearRegression()
model.fit(X3_train, y3_train)

# Test the model (optional)
y3_pred = model.predict(X3_test)
addedNumWords = MAE(y3_pred,y_test)


# In[129]:


print("MAE including number of words per utterance: ", addedNumWords)


# In[130]:


offset = []
utterance_lengths = []
durations = []
has_questions = []  # List to store 'has_q' feature
num_words = []
num_sentences = []

# Iterating through each episode
for episode_id, turns in dataset.items():
    # Iterating through each turn in the episode
    for turn in turns:
        utterance_length = len(turn['utterance'])
        duration = turn['duration']
        has_question = int(turn['has_q'])  # Convert boolean to int (True to 1, False to 0)
        words = turn['n_words']
        sentence = turn['n_sentences']
        
        offset.append(1)
        utterance_lengths.append(utterance_length)
        durations.append(duration)
        has_questions.append(has_question)
        num_words.append(words)
        num_sentences.append(sentence)
        
X4_train = np.column_stack((offset, utterance_lengths, has_questions, num_words, num_sentences))  # Combine utterance lengths and has_q features
y4_train = np.array(durations)  # Target (durations)


# In[131]:


X4_test = np.column_stack((test_offset, test_utterance_lengths, test_has_questions, test_num_words, test_num_sentences))

# Create and train the model
model = LinearRegression()
model.fit(X4_train, y4_train)

# Test the model (optional)
y4_pred = model.predict(X4_test)
addedNumSentence = MAE(y4_pred,y_test)


# In[132]:


print("MAE including sentences per utterance: ", addedNumSentence)


# In[133]:


def one_hot_encode_speakers(speaker_names, include_other=True):
    num_speakers = len(speaker_names)
    one_hot_dict = {}

    for i, speaker in enumerate(speaker_names):
        encoding = [0] * i + [1] + [0] * (num_speakers - i - 1)
        if include_other:
            encoding.append(0)  # Add a 0 for 'other'
        one_hot_dict[speaker] = encoding
    
    if include_other:
        # Add a separate entry for 'other' with its own one-hot encoding
        other_encoding = [0] * num_speakers + [1]
        one_hot_dict['other'] = other_encoding

    return one_hot_dict


# In[134]:


top100 = one_hot_encode_speakers(sorted_dataSpeaker)


# In[135]:


offset = []
utterance_lengths = []
durations = []
has_questions = []  # List to store 'has_q' feature
num_words = []
num_sentences = []
speaker_one_hot = []

# Iterating through each episode
for episode_id, turns in dataset.items():
    # Iterating through each turn in the episode
    for turn in turns:
        utterance_length = len(turn['utterance'])
        duration = turn['duration']
        has_question = int(turn['has_q'])  # Convert boolean to int (True to 1, False to 0)
        words = turn['n_words']
        sentence = turn['n_sentences']
        if turn['speaker'] in top100.keys():
            oneHot = top100[turn['speaker']]
        else:
            oneHot = top100['other']
        
        offset.append(1)
        utterance_lengths.append(utterance_length)
        durations.append(duration)
        has_questions.append(has_question)
        num_words.append(words)
        num_sentences.append(sentence)
        speaker_one_hot.append(oneHot)
        
X5_train = np.column_stack((offset, utterance_lengths, has_questions,num_words,num_sentences,speaker_one_hot))  # Combine utterance lengths and has_q features
y5_train = np.array(durations)  # Target (durations)


# In[136]:


X5_train[0]


# In[139]:


X5_test = np.column_stack((test_offset, test_utterance_lengths, test_has_questions, test_num_words, test_num_sentences, test_speaker_one_hot))

# Create and train the model
model = LinearRegression()
model.fit(X5_train, y5_train)

# Test the model (optional)
y5_pred = model.predict(X5_test)
addedSpeaker100 = MAE(y5_pred,y_test)


# In[140]:


print("MAE including a one hot for the top 100 speakers: ", addedSpeaker100)


# In[141]:


def featureCreate(length, hasq, words, sentence, speaker, dataset):
    offset = []
    utterance_lengths = []
    durations = []
    has_questions = []  # List to store 'has_q' feature
    num_words = []
    num_sentences = []
    speaker_one_hot = []

    # Iterating through each episode
    for episode_id, turns in dataset.items():
        # Iterating through each turn in the episode
        for turn in turns:
            if length:
                utterance_length = len(turn['utterance'])
                utterance_lengths.append(utterance_length)
            else:
                utterance_lengths.append(0)
            if hasq:
                has_question = int(turn['has_q'])  # Convert boolean to int (True to 1, False to 0)
                has_questions.append(has_question)
            else:
                has_questions.append(0)
            if words:
                words = turn['n_words']
                num_words.append(words)
            else:
                num_words.append(0)
            if sentence:  
                sentence = turn['n_sentences']
                num_sentences.append(sentence)
            else:
                num_sentences.append(0)
            if speaker:   
                if turn['speaker'] in top100.keys():
                    oneHot = top100[turn['speaker']]
                else:
                    oneHot = top100['other']
                speaker_one_hot.append(oneHot)
            else:
                speaker_one_hot.append(0)
            
            duration = turn['duration']
            durations.append(duration)
                
            offset.append(1)
            

    X = np.column_stack((offset, utterance_lengths, has_questions,num_words,num_sentences,speaker_one_hot))  # Combine utterance lengths and has_q features
    y = np.array(durations)  # Target (durations)
    
    return X,y


# In[142]:


def pipeline(length = True, hasq = True, words = True, sentence = True, speaker = True, normalize = False, m = False, dataset = dataset):
    X_train, y_train = featureCreate(length, hasq, words, sentence, speaker, dataset)
    X_test, y_test = featureCreate(length, hasq, words, sentence, speaker, datasetTest)
    
    if normalize:
        scaler = StandardScaler()
        y_train = scaler.fit_transform(np.log(y_train.reshape(-1, 1)))
        y_test = np.log(y_test.reshape(-1, 1))  # Log transform the test data without fitting the scaler
        

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    if normalize:
        y_pred_transformed = model.predict(X_test)
        # Inverse the standardization and the log transformation
        y_pred_standardized_inverse = scaler.inverse_transform(y_pred_transformed)
        y_pred_inverse = np.exp(y_pred_standardized_inverse)
        mae = metrics.mean_absolute_error(y_pred_inverse, np.exp(y_test))
        mse = metrics.mean_squared_error(y_pred_inverse, np.exp(y_test))
    else:
        # Test the model (optional)
        y_pred = model.predict(X_test)
        mae = MAE(y_pred,y_test)
        mse = MSE(y_pred,y_test)
        
    if m:
        pipeline = [mae, mse]
    else:
        pipeline = mae
    
    return pipeline


# In[143]:


noLength = pipeline(False,True,True,True,True,False,True)
print("Without length feature: ", noLength)

noQuestion = pipeline(True,False,True,True,True,False,True)
print("Without question feature: ", noQuestion)

noWords = pipeline(True,True,False,True,True,False,True)
print("Without words feature: ", noWords)

noSentence = pipeline(True,True,True,False,True,False,True)
print("Without sentence feature: ", noSentence)

noSpeaker = pipeline(True,True,True,True,False,False,True)
print("Without speaker feature: ", noSpeaker)


# In[144]:


justLength = pipeline(True,False,False,False,False)
print("Length: ", justLength)

justSpeakerLength = pipeline(True,False,False,False,True)
print("Speaker and Length feature: ", justSpeakerLength)

justSpeakerLengthWords = pipeline(True,False,True,False,True)
print("Speaker, length, and Words Features: ", justSpeakerLengthWords)

SpeakerLengthWordsSentence = pipeline(True,False,True,True,True)
print("Speaker, length, words, Sentence Features: ", SpeakerLengthWordsSentence)


# In[145]:


onlyWords = pipeline(False,False,True,False,False)
print("Words only: ", onlyWords)

onlySentence = pipeline(False,False,False,True,False)
print("Sentence only: ", onlySentence)


# In[146]:


SpeakerWords = pipeline(False,False,True,False,True)
print("Speaker and Words: ", SpeakerWords)

SpeakerSentence = pipeline(False,False,False,True,True)
print("Speaker and Sentence: ", SpeakerSentence)


# In[147]:


All = pipeline()
print("All features: ", All)


# ## Testing With Normalization

# In[148]:


from sklearn.preprocessing import StandardScaler
# standardizing data (currently right-skewed)
durations_log_transformed = np.log(np.array(durations))

# Standardize the log-transformed data
scaler = StandardScaler()
durations_standardized = scaler.fit_transform(durations_log_transformed.reshape(-1, 1)).flatten()

# Plot the histogram of the standardized log-transformed data
plt.hist(durations_standardized, bins=20)
plt.title('Histogram of Standardized Log-Transformed Durations')
plt.xlabel('Standardized Log(Duration)')
plt.ylabel('Frequency')
plt.show()


# In[149]:


noLength = pipeline(False,True,True,True,True,True,True)
print("Without length feature: ", noLength)

noQuestion = pipeline(True,False,True,True,True,True,True)
print("Without question feature: ", noQuestion)

noWords = pipeline(True,True,False,True,True,True,True)
print("Without words feature: ", noWords)

noSentence = pipeline(True,True,True,False,True,True,True)
print("Without sentence feature: ", noSentence)

noSpeaker = pipeline(True,True,True,True,False,True,True)
print("Without speaker feature: ", noSpeaker)


# In[150]:


justLength = pipeline(True,False,False,False,False,True,True)
print("Length: ", justLength)

justSpeakerLength = pipeline(True,False,False,False,True,True,True)
print("Speaker and Length feature: ", justSpeakerLength)

justSpeakerLengthWords = pipeline(True,False,True,False,True,True,True)
print("Speaker, length, and Words Features: ", justSpeakerLengthWords)

SpeakerLengthWordsSentence = pipeline(True,False,True,True,True,True,True)
print("Speaker, length, words, Sentence Features: ", SpeakerLengthWordsSentence)


# ## Our model VS Speechify's base of 200 words per minute

# In[151]:


def wpm(dataset):
    utterance_lengths = []
    word_counts = []
    durations = []
    predictions = []

    # Iterating through each episode
    for episode_id, turns in dataset.items():
        # Iterating through each turn in the episode
        for turn in turns:
            utterance_length = len(turn['utterance'])
            word_count = turn['n_words']
            duration = turn['utterance_end'] - turn['utterance_start']
            prediction = word_count / (200/60)
            
            utterance_lengths.append(utterance_length)
            word_counts.append(word_count)
            durations.append(duration) 
            predictions.append(prediction)
   
    return predictions


# In[152]:


print(MAE(wpm(datasetTest), test_durations))


# In[153]:


SpeakerLengthWordsSentence = pipeline(True,False,True,True,True)
print("Speaker, length, words, Sentence Features: ", SpeakerLengthWordsSentence)


# #### Our linear regression model is almost as good as Speechify's 200wpm baseline model that it uses to predict the length of speeches users give to it.

# In[ ]:




