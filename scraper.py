#!/usr/bin/python
# coding: utf-8

import re
import requests
from bs4 import BeautifulSoup
from enum import Enum, auto
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from pprint import pprint


class Scraper(object):

    def __init__(self):
        pass

    def retrieve_html(self, url):
        req = requests.get(url)
        soup = BeautifulSoup(req.content, "html.parser")
        return soup


class ResultScaraper(Scraper):
    racehead_labels = (
        "race_no",
        "tit",
        "meta",
        "title",
        "weather",
        "condition",
    )

    tit_labels = (
        "date",
        "week",
        "kai",
        "lacation",
        "nichi",
        "start_time",
    )

    meta_labels = (
        "cource",
        "clockwise",
        "distance",
    )

    score_labels = (
        "arrival_order",
        "frame_no",
        "horse_no",
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
        "passing_order_1st",
        "passing_order_2nd",
        "passing_order_3rd",
        "passing_order_4th",
    )

    def __init__(self):
        super(ResultScaraper, self).__init__()

    def extract_racehead(self, soup):
        racehead = []
        # <td>タグ
        td_tags = soup.select('div#raceTit td')
        for td_tag in td_tags:
            # <p>, <h1>タグ
            p_h1_tags = td_tag.select('p, h1')
            if p_h1_tags:
                for p_h1_tag in p_h1_tags:
                    racehead.append(p_h1_tag.get_text().strip("()\t\n\x0b\x0c\r "))
            else:
                racehead.append(td_tag.get_text().strip("()\t\n\x0b\x0c\r "))
            # <img>タグ
            img_tags = td_tag.select("img")
            if img_tags:
                for img_tag in img_tags:
                    racehead.append(img_tag['alt'])
        assert len(self.racehead_labels) == len(racehead)
        racehead_dict = OrderedDict(zip(self.racehead_labels, racehead))
        return racehead_dict

    def parse_racehead(self, racehead_dicts):
        racehead_dicts["race_no"] = re.sub(r"R", "", racehead_dicts["race_no"])
        # racehead_dicts["tit"] = self.parse_tit(racehead_dicts["tit"])
        # racehead_dict["meta"] = self.parse_meta(recehead_dicts["meta"])
        self.parse_date(racehead_dicts["date"])

        # racehead_dicts["meta"] = self._parse_meta(racehead_dicts["meta"])
        # racehead_dicts["title"] = self._parse_title(racehead_dicts["title"])
        # self._to_digit(racehead_dicts)

    def extract_scores(self, soup):
        score_dicts = []
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
            assert len(self.score_labels) == len(score)
            score_dict = OrderedDict(zip(self.score_labels, score))
            score_dicts.append(score_dict)
        return score_dicts

    def parse_scores(self, scores):
        for score in scores:
            score["time"] = self._to_sec(score["time"])
            # score["last3f_time"] = self._to_sec(score["last3f_time"])
            score["horse_info"] = self._parse_horse_info(score["horse_info"])
            score["passing_order"] = self._parse_passing_order(score["passing_order"])

            for k, v in score.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        v[_k] = self._str_to_digit(_v)
                else:
                    score[k] = self._str_to_digit(v)

        return scores

    def _parse_horse_info(self, horse_info):
        s = re.search("(牡|牝|せん)(\d+)/(\d+)\(([\+\-]?\d+)\)/(B?)", horse_info)
        if s:
            return dict(zip(self.horse_info_labels, s.groups()))
        else:
            raise

    def _parse_passing_order(self, passing_order):
        s = passing_order.split("-")
        if s:
            return dict(zip(self.passing_order_labels, s))

    def _to_sec(self, time):
        base_time = datetime.strptime("00.00.0", "%M.%S.%f")
        time = datetime.strptime(time, "%M.%S.%f")
        sec = timedelta.total_seconds(time - base_time)
        return sec

    def _str_to_digit(self, data):
        if not isinstance(data, str):
            return data

        if data == '':
            data = None
        elif data.isdigit():
            data = int(data)
        else:
            try:
                data = float(data)
            except ValueError:
                pass
        return data

    def _parse_race_no(self, data):
        return re.sub("R", "", data)

    def _parse_tit(self, data):
        s = data.split("|")
        for e in s:
            pass

    def _parse_meta(self, meta):
        pass

    def _parse_title(self, title):
        pass


def test_scraper(url):
    scraper = ResultScaraper()
    soup = scraper.retrieve_html(url)
    score_dicts = scraper.extract_scores(soup)
    score_dicts = scraper.parse_scores(score_dicts)
    return score_dicts


def test_racehead(url):
    scraper = ResultScaraper()
    soup = scraper.retrieve_html(url)
    racehead_dict = scraper.extract_racehead(soup)
    # racehead_dict = scraper.parse_racehead(racehead_dict)
    return racehead_dict


def crawl_resutl_sites():
    pass


if __name__ == "__main__":
    url = "https://keiba.yahoo.co.jp/race/result/1906030212/"

    # score_dicts = test_scraper(url)
    # pprint(score_dicts)

    racehead_dict = test_racehead(url)
    pprint(racehead_dict)
