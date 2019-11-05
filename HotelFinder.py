import datetime
import re
import numpy as np
import pandas as pd
import pygal
from flask import Flask, redirect, url_for, request, render_template
from math import sqrt
from sklearn import metrics
from sklearn.externals import joblib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from selenium.webdriver.support.ui import WebDriverWait
import re
import requests
from itertools import count
import pandas as pd
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import random
from random import uniform
from datetime import datetime
import json
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import FreqDist

from gensim.models import Phrases
from gensim.models.phrases import Phraser

from pygal.style import DarkSolarizedStyle

import numpy as np
from operator import itemgetter

def simple_get(url):

    with closing(get(url, stream=True)) as resp:
        if is_good_response(resp):
            return resp.content
        else:
            return None


def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)

def ahref_extractor(url_given):   
    url = simple_get(url_given)
    html = BeautifulSoup(url, 'html.parser')
    all_links = html.find_all("a")
    list1=[]
    for link in all_links:
        full_menu_list1 = link.get("href")
        if(full_menu_list1):
            list1.append(full_menu_list1.split('\n') )
            
    full_menu_list = list1[0]
    for i in range(1,len(list1)):
        full_menu_list += list1[i]
    links_in_list = full_menu_list    
        
    hotel_links = []
    for link in links_in_list:
        test_links = re.search(r'^(/biz/[a-z]{4}[a-z-0-9-]*)',link)
        if(test_links):
            hotel_links.append(test_links.group())
    hotel_links = list(set(hotel_links))
    return(hotel_links)

def google_to_yelp(google_typed):
  # go to google browser
    browser = webdriver.Chrome()
    browser.get('http://www.google.com')
    search = browser.find_element_by_name('q')

    # Google Search
    search.send_keys(google_typed)
    time.sleep(1)
    search.send_keys(Keys.RETURN)# hit return after you enter search text   

    # Get the current url
    Search_page_url = browser.current_url
    results = browser.find_elements_by_css_selector('div.g')

    # collecting the yelp results
    href=[]
    for result in results:
        link = result.find_element_by_tag_name("a")
        if(re.search(r'https://www.yelp.com',link.get_attribute("href"))):
            href.append(link.get_attribute("href"))         
    browser.get(href[0])    

    time.sleep(uniform(1,7)) # sleep for 5 seconds so you can see the results
    return(href[0],browser)

def DataFrame_creation_url(url,df_review_hotel):
    
    html = BeautifulSoup(url, 'html.parser')
    script = html.find_all('script')

    # finding hotel_name
    meta = html.find('meta', property="og:title")
    hotel_name = re.search(r'([A-z]{1}[\'’A-z -]+) - [A-z ]+,',meta['content']).group(1)
    # script[7] has the review information
    script[7]
    
    for j in script[7]:
        result=j

    val = re.search(r'{"aggregateRating": {"reviewCount": ([0-9]*), "@type": "AggregateRating", "ratingValue": ([0-9.]*)}',result)
    comment_raw = re.search(r'"review": \[\{"reviewRating":(.*)"name":',result)
    address = re.search(r'"addressLocality": "(.*)", "addressRegion": "(.*)", "streetAddress": "(.*)", "postalCode": "([0-9]*)"',result)
    if(address):
        place = address.group(1)
        state = address.group(2)
        exact_address = address.group(3)
        zip_code = address.group(4)
    else : 
        place = None
        state = None
        exact_address = None
        zip_code = None

    if(val):
        review_count, rating_val = val.group(1),val.group(2)
        comment = comment_raw.group(1)
    else:
        review_count, rating_val =0,0
        comment = '#empty'
        
    df_review_hotel = df_review_hotel.append({'state': state, 'county': place,'address': exact_address,'zip':zip_code, 'hotel_name': hotel_name , 'review_count': review_count , 'rating_value': rating_val, 'comment': comment}, ignore_index=True)
#     browser.quit()
    return(df_review_hotel,script[7])


def general_search_hotel_database(google_typed,df_review_hotel):
        
    href,browser = google_to_yelp(google_typed)

    # Collecting all the hotel links and store in dataframe
    hotel_links = ahref_extractor(href)
    print(hotel_links)

#     for i in range(len(hotel_links)):
    for i in range(0,5):
        
        browser.quit()
        
        # Getting all the results from yelp location page
        
        browser = webdriver.Chrome()
        browser.get("https://www.yelp.com"+hotel_links[i])
        hotel_name = re.search(r'/biz/(.*)',hotel_links[i]).group(1)

        url = simple_get("https://www.yelp.com"+hotel_links[i])
        df_review_hotel,script = DataFrame_creation_url(url,df_review_hotel)
        
    return(df_review_hotel,script)

def specific_hotel_search_database(google_typed,df_review_hotel):
    
    href,browser = google_to_yelp(google_typed)
    url = simple_get(href)
    
    df_review_hotel_final, script = DataFrame_creation_url(url,df_review_hotel)
    review_count = int(df_review_hotel_final['review_count'][0])
    if(review_count >100):
        rc = 100
    else:
        rc = review_count

    for i in range(20,rc,20):
        time.sleep(uniform(1,6))
        href1 = href+'?start='+str(i)
        url = simple_get(href1)
        df_review_hotel_1, script = DataFrame_creation_url(url,df_review_hotel)
        df_review_hotel_final = pd.concat([df_review_hotel_final,df_review_hotel_1]).reset_index(drop=True)
        
    
    return(df_review_hotel_final,script)


def extracting_costrange_cuisine(val):
    reviews = re.search(r'(.*), "servesCuisine":(.*) ',val).group(1)
    costrange_cuisine = re.search(r'(.*), "servesCuisine":(.*), ',val).group(2)
    reviews_json = json.loads(reviews)
    review_ind_rating=[]
    review_comment = []
    review_date = []
    for i in range(len(reviews_json)):
        review_ind_rating.append(reviews_json[i]['reviewRating']['ratingValue'])
        review_comment.append(reviews_json[i]['description'])
        review_date.append(reviews_json[i]['datePublished'])
    costrange_cuisine = '{"servesCuisine":'+costrange_cuisine+'}'
    costrange_cuisine_json = json.loads(costrange_cuisine)
    # tmp1
    cusine = [costrange_cuisine_json['servesCuisine']]
    costrange = [costrange_cuisine_json['priceRange']]
    
    review_df = pd.DataFrame({'review_ind_rating': review_ind_rating,
                              'review_comment': review_comment,
                              'review_date': review_date})
    
    cusine_cost_df = pd.DataFrame({'cusine': cusine,
                              'costrange': costrange})
    
    
    return(review_df, cusine_cost_df)


def review_and_total_df(df2):
    # Creating the whole data frame with cusine and costrange
    hotel_primaryKey = []
    cusine_cost_df = pd.DataFrame(columns=['cusine', 'costrange'])

    for i in range(len(df2)):

        hotel_primaryKey.append('h'+str(i))

        if(df2['review_count'].loc[i]==0):
            cusine_cost_df = cusine_cost_df.append({'cusine': None, 'costrange':None }, ignore_index=True)
        else:
            temp = '[{"reviewRating":'+df2['comment'][i]
            review_df1, cusine_cost_df1 = extracting_costrange_cuisine(temp)
            cusine_cost_df = cusine_cost_df.append(cusine_cost_df1, ignore_index=True)

    # hotel_primaryKey
    df2['cusine'] = cusine_cost_df['cusine']
    df2['cost_range'] = cusine_cost_df['costrange']
    df2['id'] = hotel_primaryKey 


    # Creating data frame for review comments
    review_df = pd.DataFrame(columns=['review_ind_rating',
                                  'review_comment',
                                  'review_date','h_id'])

    for i in range(len(df2)):
        if(df2['review_count'].loc[i]==0):
            review_df1 = review_df.append({'review_ind_rating': None, 'review_comment':None, 'review_date':None, 'h_id': df2['id'].loc[i] }, ignore_index=True)
        else:
            temp = '[{"reviewRating":'+df2['comment'][i]
            review_df1, cusine_cost_df1 = extracting_costrange_cuisine(temp)
            review_df1['h_id'] = df2['id'].loc[i] 
        review_df = review_df.append(review_df1, ignore_index = True)

    review_df['rc_id']= review_df.index
    
    return(df2,review_df)


# Text data cleaning
def text_cleaning_part1(X):
    tweets = []
    stemmer = WordNetLemmatizer()

    for s in range(0, len(X)):
        tweet = re.sub(r'\W', ' ', str(X[s]))  # specal char

        tweet = tweet.lower()  # lowercase

        tweet = re.sub(r'\s+[A-z]\s+', ' ', tweet)  # single characters like 'a,i'

        tweet = re.sub(r'\^[A-z]\s+', ' ', tweet)  # single characters in begining

        tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)  # multiple spaces

        tweet = re.sub(r'^b\s+', ' ', tweet)  # Removing prefixed 'b'

        tweet = tweet.split()  # Lemmatization

        # tweet = [stemmer.lemmatize(word) for word in tweet]
        tweet = ' '.join(tweet)

        tweets.append(tweet)

    return (tweets)


# Cleaning text
# text cleaning continution
nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, tags=['NOUN', 'ADJ', 'PROPN']):  # filter noun and adjective
    output = []
    for sent in texts:
        doc = nlp("".join(sent))
        # print("1",doc)
        review = [token.lemma_ for token in doc if token.pos_ in tags]
        review = ' '.join(review)
        output.append(review)
    return output


def remove_stopwords(rev,stop_words):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


# Removing url from tweets
def removing_url(df_v1, col_name):
    for i in range(len(df_v1[col_name])):
        a = re.search(r'([A-z0-9]+\.[.A-z0-9/]+)', df_v1[col_name][i])
        if (a):
            string = df_v1[col_name][i]
            df_v1[col_name][i] = string.replace(a.group(1), '')
    return (df_v1)


# Removing null
def removing_null(df_v1, col_name):
    null_tweets = []
    for i in range(len(df_v1)):
        if (len(df_v1[col_name][i]) == 0):
            null_tweets.append(i)
    df_v1 = df_v1.drop(null_tweets).reset_index(drop=True)
    return (df_v1)


def cleaning_text(df_v1, col_name,stop_words):
    #     df_v1 = removing_url(df_v1,col_name)
    df_v1[col_name] = text_cleaning_part1(df_v1[col_name])
    df_v1[col_name] = [remove_stopwords(r.split(),stop_words) for r in df_v1[col_name]]
    df_v1[col_name] = lemmatization(df_v1[col_name])
    # df_v1 = removing_null(df_v1,col_name)
    return (df_v1)



def unigram_converter(tweets):
    token_ = [doc.split(" ") for doc in tweets]

    return (token_)


# frequent words finder


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    #     plt.figure(figsize=(20,5))
    #     ax = sns.barplot(data=d, x= "word", y = "count")
    #     ax.set(ylabel = 'Count')
    #     plt.show()
    return (d)


def bigram_converter(review):
    token_ = [doc.split(" ") for doc in review]
    bigram = Phrases(token_, min_count=1, threshold=2, delimiter=b' ')

    bigram_phraser = Phraser(bigram)

    bigram_token = []
    for sent in token_:
        bigram_token.append(bigram_phraser[sent])

    return (bigram_token)


def freq_words_bigrams(x, terms=30):
    all_words = []
    for text in x:
        all_words += text

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)

    return (d)


def remove_nonfreq_words(rev, total_words):
    rev_new = " ".join([i for i in rev if i in total_words])
    return rev_new

def run_mainfile(google_typed):
    df_review_hotel = pd.DataFrame(
        columns=['state', 'county', 'address', 'zip', 'hotel_name', 'review_count', 'rating_value', 'comment'])
    df1,script = specific_hotel_search_database(google_typed,df_review_hotel)

    df2,review_df = review_and_total_df(df1)

    pos_review = review_df[review_df['review_ind_rating']>=4].reset_index(drop=True)
    neg_review = review_df[review_df['review_ind_rating']<4].reset_index(drop=True)
    pos_count = (len(pos_review)/len(review_df))*100
    neg_count = (len(neg_review)/len(review_df))*100
    stop_words = stopwords.words('english')

    pos_review1 = pos_review.copy()
    pos_review_clean_df = cleaning_text(pos_review1,col_name = 'review_comment',stop_words=stop_words)
    pos_words = freq_words(pos_review_clean_df['review_comment'],40)

    pos_bigram_words = bigram_converter(pos_review_clean_df['review_comment'])
    frq_words = freq_words_bigrams(x=pos_bigram_words, terms = 20)

    bi_words=[]
    one_words=[]
    for i in range(0,len(frq_words)):
        if(len(list(frq_words['word'])[i].split())==2):
            bi_words.append(list(frq_words['word'])[i])
        else:
            one_words.append(list(frq_words['word'])[i])

    most_freq_pos_words_len = [remove_nonfreq_words(list(set(r.split())), list(frq_words['word'])) for r in
                               pos_review_clean_df['review_comment']]

    index_store = []
    for i in range(len(pos_review['review_comment'])):
        if (len(most_freq_pos_words_len[i]) > 0):
            index_store.append([i, len(pos_review['review_comment'].iloc[i]) / len(most_freq_pos_words_len[i])])

    index_store = sorted(index_store, key=itemgetter(1))
    top_3_index = [i[0] for i in index_store[-20:]]
    top_sent = [pos_review['review_comment'].iloc[i] for i in top_3_index]
    from itertools import count
    cnt = count()
    n = 4
    final_sample_review = sorted(top_sent, key=lambda word: (len(word), next(cnt)), reverse=False)[:n]
    return(final_sample_review, bi_words,one_words,pos_count,neg_count)


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Hotel_search.html')

@app.route('/search/<search_key>/')
def success(search_key):

    google_typed = search_key

    and_split = google_typed.split('&')
    if (len(and_split) > 1):
        google_1_typed = and_split[1]
        final_sample_review1, bi_word1, one_word1,pos_count1,neg_count1 = run_mainfile(google_1_typed)

        # create a bar chart
        title_1 = google_1_typed
        bar_chart_1 = pygal.Bar(width=1200, height=600,
                              explicit_size=True, title=title_1, style=DarkSolarizedStyle)
        bar_chart_1.x_labels = ['pos','neg']
        bar_chart_1.add('prec', [pos_count1,neg_count1])

        google_2_typed = and_split[0] + 'in' + and_split[1].split('in')[1]
        final_sample_review2 ,bi_word2, one_word2,pos_count2,neg_count2 = run_mainfile(google_2_typed)

        title_2 = google_2_typed
        bar_chart_2 = pygal.Bar(width=1200, height=600,
                              explicit_size=True, title=title_2, style=DarkSolarizedStyle)
        bar_chart_2.x_labels = ['pos', 'neg']
        bar_chart_2.add('perc', [pos_count2, neg_count2])

        return render_template('result_page_3.html', pos_count1= pos_count1,neg_count1=neg_count1, bi_word1=bi_word1, one_word1= one_word1,len_bi_1 = len(bi_word1), len_one_1=len(one_word1),final_sample_review1=final_sample_review1 , title_1= title_1, bar_chart_1= bar_chart_1,pos_count2= pos_count2,neg_count2=neg_count2, bi_word2 = bi_word2, one_word2= one_word2,len_bi_2 = len(bi_word2), len_one_2=len(one_word2),final_sample_review2 = final_sample_review2, title_2 = title_2, bar_chart_2 = bar_chart_2)

    else:
        final_sample_review, bi_word, one_word,pos_count,neg_count = run_mainfile(google_typed)

        title = google_typed
        bar_chart = pygal.Bar(width=1200, height=600,
                                explicit_size=True, title=title, style=DarkSolarizedStyle)
        bar_chart.x_labels = ['pos', 'neg']
        bar_chart.add('perc', [pos_count, neg_count])

    return render_template('result_page.html', bi_word1=bi_word, one_word1= one_word, title_1= title, bar_chart_1= bar_chart)



@app.route('/search', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        search_key = request.form['search_key']
        print(search_key)
        return redirect(url_for('success', search_key = search_key))
    else:
        search_key = request.form['search_key']
        return redirect(url_for('success', search_key = search_key))

if __name__ == '__main__':
    app.run(debug=True)