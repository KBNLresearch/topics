#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

from bottle import abort
from bottle import get
from bottle import request
from bottle import run
from lxml import etree
from sklearn.externals import joblib

TOPICS = ['politics', 'business', 'culture', 'science', 'sports']
TYPES = ['person', 'organisation', 'location', 'other']

SOLR_URL = 'http://linksolr1.kbresearch.nl/dbpedia/select?'

def get_ocr(url):
    response = requests.get(url)
    xml = etree.fromstring(response.content)
    ocr = etree.tostring(xml, encoding='utf-8', method='text')
    return ' '.join(ocr.decode('utf-8').split())

def get_abstract(url):
    response = requests.get(SOLR_URL, params={'q': 'id:"{}"'.format(url)})
    xml = etree.fromstring(response.content)
    text = xml.find('.//str[@name="abstract"]').text
    lang = xml.find('.//str[@name="lang"]').text
    return text, lang

@get('/')
def index():
    url = request.params.get('url')
    text = request.params.get('text')
    lang = request.params.get('lang')
    content_type = request.params.get('type')

    if not url and not text:
        abort(400, 'Missing argument "url=..." or "text=...".')

    if not lang:
        lang = 'nl'

    if not content_type:
        content_type = 'news'

    if url:
        url = url.encode('latin-1').decode('utf-8')
        if 'dbpedia' in url:
            content_type = 'dbp'

    if text:
        text = text.encode('latin-1').decode('utf-8')

    if content_type == 'dbp':
        if url:
            text, lang = get_abstract(url)
        if lang == 'en':
            counts = dbp_topics_en_vct.transform([text])
            topic_probs = dbp_topics_en_clf.predict_proba(counts)[0]
            counts = dbp_types_en_vct.transform([text])
            type_probs = dbp_types_en_clf.predict_proba(counts)[0]
        else:
            counts = dbp_topics_nl_vct.transform([text])
            topic_probs = dbp_topics_nl_clf.predict_proba(counts)[0]
            counts = dbp_types_nl_vct.transform([text])
            type_probs = dbp_types_nl_clf.predict_proba(counts)[0]
    else:
        lang = 'nl'
        if url:
            text = get_ocr(url)
        counts = news_topics_nl_vct.transform([text])
        topic_probs = news_topics_nl_clf.predict_proba(counts)[0]

    result = {}
    result['lang'] = lang
    result['text'] = text
    result['type'] = content_type
    result['topics'] = {TOPICS[i]:p for (i,p) in enumerate(topic_probs)}

    if content_type == 'dbp':
        result['types'] = {TYPES[i]:p for (i,p) in enumerate(type_probs)}

    return result

if __name__ == '__main__':
    # News article topic classifier (Dutch only)
    news_topics_nl_clf = joblib.load('news_topics_nl_clf.pkl')
    news_topics_nl_vct = joblib.load('news_topics_nl_vct.pkl')

    # DBpedia topic classifier (Dutch and English)
    dbp_topics_nl_clf = joblib.load('dbp_topics_nl_clf.pkl')
    dbp_topics_nl_vct = joblib.load('dbp_topics_nl_vct.pkl')
    dbp_topics_en_clf = joblib.load('dbp_topics_en_clf.pkl')
    dbp_topics_en_vct = joblib.load('dbp_topics_en_vct.pkl')

    # DBpedia type classifier (Dutch and English)
    dbp_types_nl_clf = joblib.load('dbp_types_nl_clf.pkl')
    dbp_types_nl_vct = joblib.load('dbp_types_nl_vct.pkl')
    dbp_types_en_clf = joblib.load('dbp_types_en_clf.pkl')
    dbp_types_en_vct = joblib.load('dbp_types_en_vct.pkl')

    run(host='localhost', port=8092)

