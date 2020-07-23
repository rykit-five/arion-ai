#!/usr/bin/python
# coding:utf-8

import time
import json
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from pprint import pprint
from abc import ABCMeta, abstractmethod


class YahooCrawler:

    FETCH_INTERVAL = 1.0
    TARGET_URL = re.compile("/race/result/\d+/")
    PATROL_URL = re.compile("/directory/(horse|jockey|trainer)/\d+/")

    SCORE_LABELS = (
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

    HORSE_INFO_LABELS = (
        "horse_sex",
        "horse_age",
        "horse_weight",
        "horse_weight_diff",
        "horse_b",
    )

    HORSE_PASSING_ORDER_LABELS = (
        "passing_order1st",
        "passing_order2nd",
        "passing_order3rd",
        "passing_order4th",
    )

    def __init__(self, base_url, base_dir):
        self.base_url = base_url
        self.base_dir = base_dir

    def get_requests(self, url):
        try:
            print("retrieving... <{0}>".format(url))
            req = requests.get(url)
            req.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print("error... <{0}>: {1}".format(url, e))
        return req

    def get_soup(self, res):
        soup = BeautifulSoup(res.content, "html.parser")
        return soup

    def retrieve_urls(self, soup):
        a_tags = soup.find_all("a")
        urls = [a.get("href") for a in a_tags]
        urls = self.filter_urls(urls)
        return urls

    def filter_urls(self, urls):
        filtered_urls = []
        for url in urls:
            if re.search(self.TARGET_URL, url) or re.search(self.PATROL_URL, url):
                filtered_urls.append(self.base_url + url)
        return filtered_urls

    def extract_race_header(self, soup):
        pass

    def extract_race_scores(self, soup):
        scores = []
        table = soup.select("table#raceScore")[0]
        for tr_tag in table.select("tbody > tr"):
            score = []
            for td_tag in tr_tag.select("td"):
                # <a>タグ
                a_tags = td_tag.select("a")
                if a_tags:
                    score.append(a_tags[0].get_text().strip())
                    a_tags[0].extract()
                # <span>タグ
                span_tags = td_tag.select("span")
                if span_tags:
                    score.append(span_tags[0].get_text().strip())
                    span_tags[0].extract()
                # 上記以外
                td_text = td_tag.get_text().strip()
                if td_text:
                    score.append(td_text)
            # データを整形
            score = dict(zip(self.SCORE_LABELS, score))
            score["odds"] = self.parse_odds(score["odds"])
            score.update(self.parse_horse_info(score["horse_info"]))
            score.update(self.parse_passing_order(score["passing_order"]))
            score.pop("horse_info")
            score.pop("passing_order")
            scores.append(score)
        return scores

    def parse_odds(self, odds):
        return re.sub(re.compile("\(|\)"), "", odds)

    def parse_horse_info(self, horse_info):
        horse_infos = re.search("(牡|牝|せん)(\d+)/(\d+)\(([\+\-]?\d+)\)/(B?)", horse_info)
        if horse_infos:
            return dict(zip(self.HORSE_INFO_LABELS, horse_infos.groups()))

    def parse_passing_order(self, passing_order):
        horse_passing_orders = re.split(re.compile("-"), passing_order)
        return dict(zip(self.HORSE_PASSING_ORDER_LABELS, horse_passing_orders))

    def save_json(self, dict, filename):
        with open(filename, "w") as f:
            json.dump(dict, f, indent=4)

    def crawl(self, url, max_depth):
        req = self.get_requests(url)
        soup = self.get_soup(req)
        new_urls = self.retrieve_urls(soup)
        self.recursively_crawl(new_urls, max_depth)

    def recursively_crawl(self, urls, max_depth):
        if max_depth == 0:
            return

        new_urls = []
        for url in urls:
            req = self.get_requests(url)
            soup = self.get_soup(req)
            new_urls.extend(self.retrieve_urls(soup))
            if re.search(self.TARGET_URL, url):
                # header = self.extract_race_scores(soup)
                scores = self.extract_race_scores(soup)
                # TODO: headerとscoresを連結
                filename = self.base_dir / url + ".json"
                self.save_json(scores, filename)
            time.sleep(self.FETCH_INTERVAL)
        self.crawl(new_urls, max_depth - 1)
        return


# 抽象クラス
class Scraper(metaclass=ABCMeta):

    @abstractmethod
    def scrape(self, url):
        pass

    @abstractmethod
    def crawl(self):
        pass

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def enroll(self):
        pass


class YahooScraper(Scraper):

    HEADER = (
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

    RE_SEXAGE = re.compile("(牡|牝|せん)(\d*)")
    RE_WEIGHT = re.compile("(\d*)\(([\+\-]?\d*)|\s\-\s\)")
    RE_B = re.compile("(B?)")

    def __init__(self):
        pass

    def scrape(self, url):
        req = None
        try:
            req = requests.get(url)
            req.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(e)
        soup = BeautifulSoup(req.content, "html.parser")
        race_scores = self.extract_race_scores(soup)
        self.str2race_scores(race_scores)
        return race_scores

    def crawl(self):
        pass

    def extract(self):
        pass

    def enroll(self):
        pass

    def str2race_scores(self, race_scores):
        for race_score in race_scores:
            race_score["arrival_order"] = int(race_score["arrival_order"])
            race_score["frame_num"] = int(race_score["frame_num"])
            race_score["horse_num"] = int(race_score["horse_num"])
            race_score["horse_info"] = self.parse_horse_info(race_score["horse_info"])
            race_score["time"] = str(race_score["time"])
            race_score["last3f_time"] = float(race_score["last3f_time"])
            race_score["passing_order"] = self.parse_passing_order(race_score["passing_order"])
            race_score["jockey_weight"] = float(race_score["jockey_weight"])
            race_score["odds"] = self.clean_string(race_score["odds"])
            race_score["popularity"] = int(race_score["popularity"])
        return race_scores

    def extract_race_scores(self, soup):
        race_scores = []
        table = soup.select("table#raceScore")[0]

        # 出馬表の各列からデータを抽出
        for tr_tag in table.select("tbody > tr"):
            score = []
            for td_tag in tr_tag.select("td"):
                # <a>タグ
                a_tags = td_tag.select("a")
                if a_tags:
                    score.append(a_tags[0].get_text().strip())
                    a_tags[0].extract()
                # <span>タグ
                span_tags = td_tag.select("span")
                if span_tags:
                    score.append(span_tags[0].get_text().strip())
                    span_tags[0].extract()
                # その他
                td_text = td_tag.get_text().strip()
                if td_text:
                    score.append(td_text)

            # 出馬表の各列のデータを辞書化
            race_scores.append(dict(zip(self.HEADER, score)))

        return race_scores

    def parse_horse_info(self, horse_info):
        ret = {
            "horse_sex": "N/A",
            "horse_age": "N/A",
            "horse_weight": "N/A",
            "horse_dweight": "N/A",
            "horse_B": "N/A",
        }

        horse_info = horse_info.split("/")
        s = re.search(self.RE_SEXAGE, horse_info[0])
        if s:
            ret["horse_sex"] = s.group(1)
            ret["horse_age"] = s.group(2)
        s = re.search(self.RE_WEIGHT, horse_info[1])
        if s:
            ret["horse_weight"] = s.group(1)
            ret["horse_dweight"] = s.group(2)
        s = re.search(self.RE_B, horse_info[2])
        if s:
            ret["horse_B"] = s.group(1)

        return ret

    def parse_passing_order(self, passing_order):
        ret = {
            "passing_order_1st": "N/A",
            "passing_order_2nd": "N/A",
            "passing_order_3rd": "N/A",
            "passing_order_4th": "N/A",
        }

        passing_order = passing_order.split("-")
        ret["passing_order_1st"] = passing_order[0]
        ret["passing_order_2nd"] = passing_order[1]
        ret["passing_order_3rd"] = passing_order[2]
        ret["passing_order_4th"] = passing_order[3]

        return ret

    def clean_string(self, s):
        s = re.sub("\s", "", s)
        s = re.sub("\(", "", s)
        s = re.sub("\)", "", s)
        return s

    def check_arrival_order(self, arrival_order):
        try:
            assert 1 < arrival_order < 18, "error: arrival order"
        except AssertionError as e:
            print(e)

    def check_frame_num(self, frame_num):
        try:
            assert 1 < frame_num < 8, "error: frame number"
        except AssertionError as e:
            print(e)

    def check_horse_num(self, horse_num):
        try:
            assert 1 < horse_num < 18, "error: horse number"
        except AssertionError as e:
            print(e)

    def check_horse_name(self, horse_name):
        try:
            assert isinstance(horse_name) == str, "error: horse name"
        except AssertionError as e:
            print(e)

    def check_horse_info(self, horse_info):
        pass

    def check_arrival_diff(self, arrival_diff):
        try:
            assert isinstance(arrival_diff) == str, "error: arrival difference"
        except AssertionError as e:
            print(e)

    def check_time(self, time):
        try:
            assert isinstance(time) == float, "error: time"
        except AssertionError as e:
            print(e)

    def chech_last3f_time(self, last3f_time):
        try:
            assert 0.0 < last3f_time < 100.0, "error: last 3F time"
        except AssertionError as e:
            print(e)

    def check_passing_order(self, passing_order):
        pass

    def check_jockey_name(self, jockey_name):
        try:
            assert isinstance(jockey_name) == str, "error: jockey name"
        except AssertionError as e:
            print(e)

    def check_jockey_weight(self, jockey_weight):
        try:
            assert 0.0 < jockey_weight < 100.0, "error: jockey weight"
        except AssertionError as e:
            print(e)

    def chech_odds(self, odds):
        try:
            assert 0.0 < odds < 100000.0, "error: odds"
        except AssertionError as e:
            print(e)

    def check_popularity(self, polularity):
        try:
            assert 0.0 < polularity < 100000.0, "error: popularity"
        except AssertionError as e:
            print(e)

    def check_trainer_name(self, trainer_name):
        try:
            assert isinstance(trainer_name) == str, "error: trainer name"
        except AssertionError as e:
            print(e)


def test_scrape():
    url = "https://keiba.yahoo.co.jp/race/result/1907040211/"  # チャンピオンズカップ
    ys = YahooScraper()
    scores = ys.scrape(url)
    pprint(scores)


if __name__ == "__main__":
    # base_url = "https://keiba.yahoo.co.jp"
    # root_url = "https://keiba.yahoo.co.jp/race/denma/1808050211/"
    # current_dir = Path.cwd()
    # dataset_dir = current_dir / "dataset"
    #
    # ys = YahooScraper(base_url, current_dir)
    # ys.crawl(root_url, max_depth=2)

    test_scrape()


