#!/usr/bin/python
#-*-coding: utf-8-*-

import os
import sys
import re
import lxml
import urllib
from bs4 import BeautifulSoup
from pprint import pprint


TAG = re.compile(r'<[^>]*?>')
SPAN_TAG = re.compile(r'<span class="scdItem">')
NEWLINE = re.compile(r'\n+')
SPACE = re.compile(r'\s+')

class YahooKeibaScraper(object):

    def __init__(self, base_url):
        self.base_url = base_url

    def fetch_url(self, url):
        try:
            response = urllib.request.urlopen(url)
            soup = BeautifulSoup(response.read(), 'lxml')
        except urllib.error.HTTPError:
            print('could not find url:', url)
            raise

        # extract race information and race results
        race_info = self._extract_race_info(soup)
        race_rslts = self._extract_race_rslts(soup)

        # test
        data = self._horse_info(race_rslts[0]['horse_info'])
        pprint(data)

        return race_info, race_rslts

    def _extract_race_info(self, soup):
        labels = ('race_no',
                  'date',
                  'race_schedule',
                  'stating_time',
                  'race_title',
                  'race_detail',
                  'weather',
                  'condition',
                  )
        race_info = soup.find('div', {'id': 'raceTit'})
        race_img = race_info.find_all('img')
        race_info = race_info.text.split('|')[: 3]
        race_info.append(race_img[0]['alt'])
        race_info.append(race_img[1]['alt'])
        race_info = [e for r in race_info for e in re.split(NEWLINE, r)]
        race_info = [re.sub(NEWLINE, '', r.strip()) for r in race_info]
        race_info = list(filter(lambda x: len(x), race_info))
        race_info = dict(zip(labels, race_info))
        return race_info

    def _extract_race_rslts(self, soup):
        labels = ('arrival_order',
                  'frame_no',
                  'horse_no',
                  'horse',
                  'horse_info',
                  'time',
                  'goal_diff',
                  'passing_order',
                  'last3f_time',
                  'jockey',
                  'jockey_weight',
                  'popularity',
                  'odds',
                  'trainer',
                  )
        denmas = soup.find('table', {'id': 'raceScore', 'class': 'dataLs mgnBS'})
        denmas = denmas.find('tbody')
        denmas = denmas.find_all('tr')
        race_rslts = []
        for denma in denmas:
            cols = denma.find_all('td')
            cols = [e for c in cols for e in re.split(SPAN_TAG, str(c))]
            cols = [re.sub(TAG, '', c) for c in cols]
            cols = [re.sub(NEWLINE, '', c) for c in cols]
            cols = dict(zip(labels, cols))
            race_rslts.append(cols)
        return race_rslts

    def _date(self, date):
        labels = ('date',
                  'week_of_day')
        date = re.sub(re.compile(r'）'), '', date)
        data = re.split(re.compile(r'（'), date)
        return dict(zip(labels, data))

    def _race_title(self, race_title):
        labels = ('title',
                  'grade')
        race_title = re.sub(re.compile(r'）'), '', race_title)
        data = re.split(re.compile(r'（'), race_title)
        return dict(zip(labels, data))

    def _race_detail(self, race_detail):
        labels = ('surface',
                  'course',
                  'length')
        data = re.split(re.compile(r'・|\s'), race_detail)
        return dict(zip(labels, data))

    def _horse_info(self, horse_info):
        labels = ('sex_age',
                  'horse_weight',
                  'weitht_diff',
                  'blinker')
        horse_info = re.sub(re.compile(r'\)'), '', horse_info)
        data = re.split(re.compile(r'/|\('), horse_info)
        return dict(zip(labels, data))

    def _starting_time(self, starting_time):
        return re.sub(re.compile(r'発走'), '', starting_time)

    def _odds(self, odds):
        return re.sub(re.compile(r'\(|\)'), '', odds)

    def _jockey(self, jockey):
        return re.sub(SPACE, '', jockey)

    def _trainer(self, trainer):
        return re.sub(SPACE, '', trainer)


if __name__ == '__main__':
    base_url = 'https://keiba.yahoo.co.jp'
    url = 'https://keiba.yahoo.co.jp/race/result/1605010810/'
    scraper = YahooKeibaScraper(base_url)
    race_info, race_rslts = scraper.fetch_url(url)

    pprint(race_info)
    pprint(race_rslts)
