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
    return xml.find('.//str[@name="abstract"]').text

@get('/')
def index():
    url = request.params.get('url')
    text = request.params.get('text')
    content_type = request.params.get('type')

    if not url and not text:
        abort(400, 'Missing argument "url=..." or "text=...".')

    if url:
        url = url.encode('latin-1').decode('utf-8')
        if url.startswith('http://nl.dbpedia.org/'):
            content_type = 'dbp'

    if text:
        text = text.encode('latin-1').decode('utf-8')

    print(content_type)

    if content_type == 'dbp':
        if url:
            text = get_abstract(url)
        counts = dbp_topics_vct.transform([text])
        topic_probs = dbp_topics_clf.predict_proba(counts)[0]
    else:
        if url:
            text = get_ocr(url)
        counts = topics_vct.transform([text])
        topic_probs = topics_clf.predict_proba(counts)[0]

    result = {}
    result['text'] = text
    result['topics'] = {TOPICS[i]:p for (i,p) in enumerate(topic_probs)}
    # result['types'] = {TYPES[i]:p for (i,p) in enumerate(type_probs)}

    return result

if __name__ == '__main__':
    topics_clf = joblib.load('topics_clf.pkl')
    topics_vct = joblib.load('topics_vct.pkl')
    dbp_topics_clf = joblib.load('dbp_topics_clf.pkl')
    dbp_topics_vct = joblib.load('dbp_topics_vct.pkl')
    # dbp_type_clf =
    # dbp_type_vct =

    run(host='localhost', port=8092)

