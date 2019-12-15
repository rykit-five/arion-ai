#!/usr/bin/python
# coding: utf-8

import re
import requests
from bs4 import BeautifulSoup
from pprint import pprint
from enum import Enum, auto
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import sqlite3
from sqlite3 import Error


class Scraper(metaclass=ABCMeta):

    def __init__(self):
        # TODO: get_req, conv_req_to_soupを統合
        pass

    def get_req(self, url):
        req = requests.get(url)
        return req

    def conv_req_to_soup(self, req):
        soup = BeautifulSoup(req.content, "html.parser")
        return soup

    # @abstractmethod
    # def extract_data(self, soup):
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def parse_data(self, soup):
    #     raise NotImplementedError


class ResultScaraper(Scraper):

    score_labels = (
        "arrival_order",
        "frame_num",
        "horse_num",
        "horse_name",
        "horse_info",
        "arrival_diff",
        "time",
        "last3f_time",
        "passing_order",
        "jockey_name",
        "jockey_weight",
        "odds",
        "popularity",
        "trainer_name",
    )

    horse_info_labels = (
        "horse_sex",
        "horse_age",
        "horse_weight",
        "horse_weight_diff",
        "horse_b",
    )

    passing_order_labels = (
        "passing_order1st",
        "passing_order2nd",
        "passing_order3rd",
        "passing_order4th",
    )

    def __init__(self):
        super(ResultScaraper, self).__init__()

    def extract_race_header(self, soup):
        pass

    def extract_scores(self, soup):
        scores = []
        table = soup.select("table#raceScore")[0]
        for tr_tag in table.select("tbody > tr"):
            score = []
            for td_tag in tr_tag.select("td"):
                # <a>タグ
                a_tags = td_tag.select("a")
                if a_tags:
                    score.append(a_tags[0].get_text().strip("()\t\n\x0b\x0c\r "))
                    a_tags[0].extract()
                # <span>タグ
                span_tags = td_tag.select("span")
                if span_tags:
                    score.append(span_tags[0].get_text().strip("()\t\n\x0b\x0c\r "))
                    span_tags[0].extract()
                # 上記以外
                td_text = td_tag.get_text().strip()
                if td_text:
                    score.append(td_text)
            # データを保存
            score = OrderedDict(zip(self.score_labels, score))
            scores.append(score)
        return scores

    def parse_scores(self, scores):
        for score in scores:
            score["time"] = self._parse_time(score["time"])
            # score.update(self._parse_horse_info(score["horse_info"]))
            # score.update(self._parse_passing_order(score["passing_order"]))
            score = self._parse_digit(score)
        return scores

    def _parse_horse_info(self, horse_info):
        s = re.search("(牡|牝|せん)(\d+)/(\d+)\(([\+\-]?\d+)\)/(B?)", horse_info)
        if s:
            return OrderedDict(zip(self.horse_info_labels, s.groups()))
        else:
            raise

    def _parse_passing_order(self, passing_order):
        passing_orders = re.split("-", passing_order)
        if passing_order:
            return OrderedDict(zip(self.passing_order_labels, passing_orders))
        else:
            raise

    def _parse_time(self, time):
        times = time.split(".")
        sec = "{}.{}".format(int(times[0]) * 60 + int(times[1]), int(time[2]))
        return sec

    def _parse_digit(self, score):
        for k, v in score.items():
            if v.isdigit():
                score[k] = int(v)
            elif v == '':
                score[k] = None
            else:
                try:
                    score[k] = float(v)
                except ValueError:
                    pass
        return score


class Register:

    def __init__(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
        self.conn = conn
        self.curs = self.conn.cursor()

    def create(self, reset=False):
        if reset:
            self.curs.execute('''DROP TABLE IF EXISTS scores''')
        self.curs.execute('''CREATE TABLE IF NOT EXISTS scores (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                arrival_order INTEGER NOT NULL,
                                frame_num INTEGER NOT NULL,
                                horse_num INTEGER NOT NULL, 
                                horse_name TEXT NOT NULL, 
                                horse_info TEXT NOT NULL,
                                arrival_diff TEXT,
                                time REAL NOT NULL,
                                last3f_time REAL NOT NULL,
                                passing_order TEXT NOT NULL,
                                jockey_name TEXT NOT NULL,
                                jockey_weight REAL NOT NULL,
                                odds REAL NOT NULL,
                                popularity REAL NOT NULL,
                                trainer_name TEXT NOT NULL
                             )''')
        self.conn.commit()

    def close(self):
        self.curs.close()
        self.conn.close()

    def insert(self, score):
        sql = '''INSERT INTO scores (
                    arrival_order,
                    frame_num,
                    horse_num,
                    horse_name,
                    horse_info,
                    arrival_diff,
                    time,
                    last3f_time,
                    passing_order,
                    jockey_name,
                    jockey_weight,
                    odds,
                    popularity,
                    trainer_name
                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
        self.curs.execute(sql, score)
        self.conn.commit()


if __name__ == "__main__":
    url = "https://keiba.yahoo.co.jp/race/result/1906030211/"
    scraper = ResultScaraper()
    request = scraper.get_req(url)
    soup = scraper.conv_req_to_soup(request)
    scores = scraper.extract_scores(soup)
    scores = scraper.parse_scores(scores)
    # pprint(scores)

    db_file = "test_db.sqlite"
    reg = Register(db_file)
    reg.create(reset=True)
    for score in scores:
        # pprint(score.values())
        reg.insert(tuple(score.values()))


