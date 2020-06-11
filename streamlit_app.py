#! /nfs/orto/home/pradhyum/.conda/envs/my_root/bin/python
import sys
import streamlit as st
import tweepy
import json
import torch
import random
import re
import urllib.request
from PIL import Image
sys.path.append('/nfs/site/disks/gia_analytics_shared/paddy/projects/git_repos/simpletransformers/')
from simpletransformers.language_modeling import LanguageModelingModel
from simpletransformers.language_generation import LanguageGenerationModel
import logging
import os
import SessionState
proxy = 'https://pradhyum:Paddy-_0)9(@proxy-chain.intel.com:911'
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

def collect_tweets(api, handle):
    # initialize a list to hold all the tweepy Tweets & list with no retweets
    new_tweets = api.user_timeline(
    screen_name=handle, tweet_mode='extended', count=200)
    alltweets = []
    # make initial request for most recent tweets with extended mode enabled to get full tweets
    # save most recent tweets
    alltweets.extend(new_tweets)
    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    # check we cannot get more tweets
    no_tweet_count = 0
    # keep grabbing tweets until the api limit is reached
    while True:
        #print(f'getting tweets before id {oldest}')
        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(
            screen_name=handle, tweet_mode='extended', count=200, max_id=oldest)
        # stop if no more tweets (try a few times as they sometimes eventually come)
        if not new_tweets:
            no_tweet_count +=1
        else:
            no_tweet_count = 0
        if no_tweet_count > 2: break
        # save most recent tweets
        alltweets.extend(new_tweets)
        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

    ## Create a dataset from downloaded tweets
    
    class user_handle:
        'Get a user handle and cache it to avoid calling too much twitter api.'
        handles = {}
        def get(handle):
            if handle not in user_handle.handles.keys():            
                try:
                    user_handle.handles[handle] = api.get_user(handle).name
                except:
                    user_handle.handles[handle] = None
            return user_handle.handles[handle]

    def replace_handle(word):
        'Replace user handles, remove "@" and "#"'
        if word[0] == '@':
            handle = re.search('^@(\w)+', word)
            if handle:
                user = user_handle.get(handle.group())
                if user is not None: return user + word[handle.endpos:]
        return word

    def keep_tweet(tweet):
        'Return true if not a retweet'
        if hasattr(tweet, 'retweeted_status'):
            return False
        return True

    def curate_tweets(tweets):
        curated_tweets = []
        for tweet in tweets:
            if keep_tweet(tweet):
                curated_tweets.append(' '.join(replace_handle(w) for w in tweet.full_text.split()))
        return curated_tweets

    curated_tweets = curate_tweets(alltweets)
    st.success(f'Total number of tweets: {len(alltweets)}\nCurated tweets: {len(curated_tweets)}')
    return curated_tweets

def process_tweets(curated_tweets):  
    def cleanup_tweet(tweet):
        "Clean tweet text"
        text = ' '.join(t for t in tweet.split() if 'http' not in t)
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        if text.split() and text.split()[0] == '.':
             text = ' '.join(text.split()[1:])
        return text

    def boring_tweet(tweet):
        "Check if this is a boring tweet"
        boring_stuff = ['http', '@', '#', 'thank', 'thanks', 'I', 'you']
        if len(tweet.split()) < 3:
            return True
        if all(any(bs in t.lower() for bs in boring_stuff) for t in tweet):
            return True
        return False

    clean_tweets = [cleanup_tweet(t) for t in curated_tweets]
    cool_tweets = [tweet for tweet in clean_tweets if not boring_tweet(tweet)]
    st.write(f'Curated tweets: {len(curated_tweets)}\nCool tweets: {len(cool_tweets)}')
    return cool_tweets

#We split data into training and validation sets (90/10).
def prepare_dataset(cool_tweets,handle):
    
    # shuffle data
    random.shuffle(cool_tweets)

    # fraction of training data
    split_train_valid = 0.9

    # split dataset
    train_size = int(split_train_valid * len(cool_tweets))
    valid_size = len(cool_tweets) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(cool_tweets, [train_size, valid_size])
    train_data_path = f'./tweets_cache/{handle}_train.txt'
    valid_data_path = f'./tweets_cache/{handle}_valid.txt'
    st.info(f'Prepared data for model input. Writing training data to {train_data_path}')
    with open(train_data_path, 'w') as f:
        f.write('\n'.join(train_dataset))

    with open(valid_data_path, 'w') as f:
        f.write('\n'.join(valid_dataset))
    return


def build_language_model(handle):
    train_args = {
        "output_dir": f"gpt2_outputs/{handle}/",
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "fp16": False,
        "train_batch_size": 32,
        "eval_batch_size":32,
        "num_train_epochs": 3,
        "tensorboard_dir": 'gpt2_tweet_runs/',
        'mlm':False,
        'use_multiprocessing':False
    }


    model = LanguageModelingModel('gpt2', 'gpt2', args=train_args,use_cuda=False)
    st.info('Training model. This may take a few mins - you may want to check back later.')
    model.train_model(f"./tweets_cache/{handle}_train.txt", eval_file=f"./tweets_cache/{handle}_valid.txt")
    return
    
    
    
def generate_tweet(handle,prompt='I think that'):
    gen_args={'length':200,
             'k':10}
    model = LanguageGenerationModel("gpt2", f"gpt2_outputs/{handle}",use_cuda=False, args=gen_args)
    generated_text = model.generate(prompt,verbose=False)
    return generated_text
    #st.write(f'Here\'s what @{handle} would have to say')
    #st.text(generated_text[0])
    
def huggingtweets(api, handle,session_state):
    st.write(f'Looking for user: @{handle}')
    #try:
    new_tweets = api.user_timeline(screen_name=handle, tweet_mode='extended', count=200)
    st.success(f'Found User @{handle}. Collecting tweets.')
    #except:
    #st.error(f'Could not find user @{handle}. Please ensure the twitter handle you entered is accurate.')

    if not os.path.exists(f'gpt2_outputs/{handle}'):
        os.mkdir(f'gpt2_outputs/{handle}')
        train_data_path = f'./tweets_cache/{handle}_train.txt'
        valid_data_path = f'./tweets_cache/{handle}_valid.txt'

        if not os.path.exists(train_data_path) and not os.path.exists(valid_data_path):
            curated_tweets = collect_tweets(api,handle)
            st.info('We download latest tweets associated to a user account through [Tweepy](http://docs.tweepy.org/).')
            cool_tweets = process_tweets(curated_tweets)
            prepare_dataset(cool_tweets,handle)
        else:
            st.success('Tweet data available from cache')

        build_language_model(handle)
    else:
        st.success(f'Model already exists for user @{handle}. Using it to generate tweet.')
    
    prompt = st.text_input(f'Give @{handle} a topic to tweet about')
    session_state.topic = st.button("Tweet")
    if session_state.topic:
        text = generate_tweet(handle,f'I want to talk about {prompt} today,')
        st.markdown(text)
        st.balloons()
    
def main():
    with open('credentials.json','r') as f:
        credentials = json.load(f)
    auth = tweepy.OAuthHandler(credentials.get('api_key'), credentials.get('api_secret'))
    api = tweepy.API(auth)

    st.title('Huggingtweets!')
    st.header('Tweet like you favorite account! Use this app to compose a tweet in the stylings of your twitter personality.')
    
    session_state = SessionState.get(handle="", button_handle_submit=False)
    session_state.handle = st.text_input('Type in the twitter handle of your choice')
    button_handle_submit = st.button("Submit")

    if button_handle_submit:
        session_state.button_handle_submit = True
#    handle = st.text_input('Type in the twitter handle of your choice')
    if session_state.button_handle_submit:
        huggingtweets(api,session_state.handle,session_state)
    #if st.button('Try another',key='restart'):
        #main()
       

if __name__ == "__main__":
    main()