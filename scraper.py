#!/usr/bin/python
# coding: utf-8

import re
import requests
from bs4 import BeautifulSoup
from enum import Enum, auto
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
from pprint import pprint
import numpy as np

from crawler import Crawler
from utils import Utils


# class Scraper(object):
#
#     def __init__(self):
#         pass
#
#     def retrieve_html(self, url):
#         req = requests.get(url)
#         req.raise_for_status()
#         soup = BeautifulSoup(req.content, "html.parser")
#         return soup


class DirectoryHorseScraper():

    horse_labels = (
        "horse_name",
        "sex",
        "trainer_name",
        "birthday",
        "hair_color",
        "owner",
        "farmer",
        "origin",
        "blood_male",
        "blood_female",
    )

    def __init__(self, soup):
        """
        コンストラクタ
        Crawlerクラスによって取得したsoupをフィールド変数に設定
        :param soup: BeautifulSoup obj: html source
        """
        self.soup = soup

    def extract(self):
        pass

    def parse(self):
        pass


class DirectoryJockyScraper():

    def __init__(self, soup):
        """
        コンストラクタ
        Crawlerクラスによって取得したsoupをフィールド変数に設定
        :param soup: BeautifulSoup obj: html source
        """
        self.soup = soup

    def extract(self):
        pass

    def parse(self):
        pass


class DirectoryTrainerScraper():

    def __init__(self, soup):
        """
        コンストラクタ
        Crawlerクラスによって取得したsoupをフィールド変数に設定
        :param soup: BeautifulSoup obj: html source
        """
        self.soup = soup

    def extract(self):
        pass

    def parse(self):
        pass


class RaceResultScaraper():

    racehead_labels = (
        "round",
        "racetitday",
        "head",
        "racetitmeta",
        "weather",
        "ground",
    )

    racetitday_labels = (
        "date",
        "week",
        "month",
        "lacation",
        "day",
        "start_time",
    )

    head_labels = (
        "title",
        "grade",
    )

    racetitmeta_labels = (
        "surface",
        "clockwise",
        "distance",
        "age",
        "class",
        "reward_1st",
        "reward_2nd",
        "reward_3rd",
        "reward_4th",
        "reward_5th",
    )

    score_labels = (
        "arrival_order",
        "frame_no",
        "horse_no",
        "horse_name",
        "horse_info",
        "arrival_diff",
        "time",
        "last_3f_time",
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
        "horse_blinker",
    )

    passing_order_labels = (
        "passing_order_1st",
        "passing_order_2nd",
        "passing_order_3rd",
        "passing_order_4th",
    )

    def __init__(self, soup):
        """
        コンストラクタ
        Crawlerクラスによって取得したsoupをフィールド変数に設定
        :param soup: BeautifulSoup obj: html source
        """
        # super(RaceResultScaraper, self).__init__()
        self.soup = soup

    def extract_racehead(self):
        """
        html(soup)からレース情報を抽出
        :return: data_dict: dict
        """
        data_list = []
        # <td>タグ
        td_tags = self.soup.select('div#raceTit td')
        for td_tag in td_tags:
            # <p>, <h1>タグ
            p_h1_tags = td_tag.select('p, h1')
            if p_h1_tags:
                for p_h1_tag in p_h1_tags:
                    data_list.append(p_h1_tag.get_text().strip("()\t\n\x0b\x0c\r "))
            else:
                data_list.append(td_tag.get_text().strip("()\t\n\x0b\x0c\r "))
            # <img>タグ
            img_tags = td_tag.select("img")
            if img_tags:
                for img_tag in img_tags:
                    data_list.append(img_tag['alt'])

        assert len(self.racehead_labels) == len(data_list)
        data_dict = dict(zip(self.racehead_labels, data_list))
        return data_dict

    def parse_racehead(self, data_dict):
        """
        data_dictの各データを整形してdata_dictを一次元化
        :param data_dict: dict
        """
        data_dict["round"] = self._parse_round(data_dict["round"])
        data_dict.update(self._parse_racetitday(data_dict["racetitday"]))
        del data_dict["racetitday"]
        data_dict.update(self._parse_head(data_dict["head"]))
        del data_dict["head"]
        data_dict.update(self._parse_racetitmeta(data_dict["racetitmeta"]))
        del data_dict["racetitmeta"]

    def extract_scores(self):
        """
        html(soup)から各馬の結果情報を抽出
        :return: data_dicts: list of dict
        """
        data_dicts = []
        table = self.soup.select("table#raceScore")[0]
        for tr_tag in table.select("tbody > tr"):
            data_list = []
            for td_tag in tr_tag.select("td"):
                # <a>タグ
                a_tags = td_tag.select("a")
                if a_tags:
                    data_list.append(a_tags[0].get_text().strip("()\t\n\x0b\x0c\r "))
                    a_tags[0].extract()
                # <span>タグ
                span_tags = td_tag.select("span.scdItem")
                if span_tags:
                    data_list.append(span_tags[0].get_text().strip("()\t\n\x0b\x0c\r "))
                    span_tags[0].extract()
                # 上記以外
                td_text = td_tag.get_text().strip()
                if td_text:
                    data_list.append(td_text)
            # TODO: parseに移動させたい
            # データを保存
            if data_list[0] in ["中止", "除外", "取消"]:
                continue

            assert len(self.score_labels) == len(data_list)
            data_dict = dict(zip(self.score_labels, data_list))
            data_dicts.append(data_dict)
        return data_dicts

    def parse_scores(self, data_dicts):
        """
        data_dicts > data_dictの各データを整形してdata_dictを一次元化
        :param data_dicts: list of dict
        """
        for data_dict in data_dicts:
            data_dict["arrival_order"] = Utils.str_to_int_or_float(data_dict["arrival_order"])
            data_dict["frame_no"] = Utils.str_to_int_or_float(data_dict["frame_no"])
            data_dict["horse_no"] = Utils.str_to_int_or_float(data_dict["horse_no"])
            data_dict["time"] = Utils.str_to_sec(data_dict["time"])
            data_dict["last_3f_time"] = Utils.str_to_int_or_float(data_dict["last_3f_time"])
            data_dict["jockey_weight"] = Utils.str_to_int_or_float(data_dict["jockey_weight"])
            data_dict["odds"] = Utils.str_to_int_or_float(data_dict["odds"])
            data_dict["popularity"] = Utils.str_to_int_or_float(data_dict["popularity"])

            data_dict.update(self._parse_horse_info(data_dict["horse_info"]))
            del data_dict["horse_info"]
            data_dict.update(self._parse_passing_order(data_dict["passing_order"]))
            del data_dict["passing_order"]

    def _parse_horse_info(self, horse_info):
        s = re.search(re.compile("(牡|牝|せん)(\d+)/(\d+)\(([\+\-]?\d+| \- )\)/(B?)"), horse_info)
        if s:
            parsed_elems = [Utils.str_to_int_or_float(_) for _ in s.groups()]
            return dict(zip(self.horse_info_labels, parsed_elems))
        else:
            raise

    def _parse_passing_order(self, data):
        """
        経過順位を格納したdata(str)をセクションごとに分解
        対応する経過順位がないセクションは-1を格納
        :param data: str
        :return: dict
        """
        parsed_elems = []

        s = data.split("-")
        if len(s) == 1 and s[0] == "":
            parsed_elems = [-1] * 4
        elif 2 <= len(s) < 4:
            parsed_elems = s + [-1] * (4 - len(s))
        parsed_elems = [Utils.str_to_int_or_float(_) for _ in parsed_elems]
        return dict(zip(self.passing_order_labels, parsed_elems))

    def _parse_round(self, data):
        """
        レース番号を表した文字列(data)からRを削除して整形
        :param data: str
        :return: int
        """
        data = re.sub(r"R", "", data)
        return int(data)

    def _parse_racetitday(self, data):
        """
        raceTitDayを表した文字列(data)をデータ別に分割して整形
        :param data: str
        :return: dict
        """
        splited_data = data.split("|")
        # todo: 例外処理
        if len(splited_data) != 3:
            raise

        parsed_elems = []
        s = re.search(re.compile("(\d{4}年\d{1,2}月\d{1,2}日)（(日|月|火|水|木|金|土)）"), splited_data[0])
        if s:
            parsed_elems.append(Utils.str_to_date(s.group(1)))
            parsed_elems.append(s.group(2))

        s = re.search(re.compile("(\d{1,2})回(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)(\d{1,2})日"), splited_data[1])
        if s:
            parsed_elems.append(Utils.str_to_int_or_float(s.group(1)))
            parsed_elems.append(s.group(2))
            parsed_elems.append(Utils.str_to_int_or_float(s.group(3)))

        s = re.search(re.compile("(\d{1,2}:\d{1,2})発走"), splited_data[2])
        if s:
            parsed_elems.append(s.group(1))
        return dict(zip(self.racetitday_labels, parsed_elems))

    def _parse_head(self, data):
        """
        レース基本情報を表した文字列(data)をデータ別に分割して整形
        :param data: str
        :return: dict
        """
        parsed_elems = []
        s = re.sub(re.compile("第\d{1,3}回"), "", data)
        s = re.search(re.compile("([^（]+)([0-9a-zA-Z（）]*)"), s)
        if s and len(s.groups()) == 2:
            parsed_elems.append(s.group(1))
            parsed_elems.append(re.sub("（|）", "", s.group(2)))
        return dict(zip(self.head_labels, parsed_elems))

    def _parse_racetitmeta(self, data):
        """
        raceTitMetaを表した文字列(data)をデータ別に分割して整形
        :param data: str
        :return: dict
        """
        parsed_elems = []

        splited_data = data.split("|")
        # todo: 例外処理
        if len(splited_data) != 7:
            raise

        s = re.search(re.compile("(芝|ダート)・(左|右)\s(\d{3,4})m"), splited_data[0])
        if s:
            parsed_elems.append(s.group(1))
            parsed_elems.append(s.group(2))
            parsed_elems.append(Utils.str_to_int_or_float(s.group(3)))
        parsed_elems.append(splited_data[3].strip())
        parsed_elems.append(splited_data[4].strip())
        s = re.sub("本賞金：|万円", "", splited_data[5])
        s = s.split("、")
        if len(s) == 5:
            s = [Utils.str_to_int_or_float(_) for _ in s]
            parsed_elems.extend(s)
        else:
            raise

        return dict(zip(self.racetitmeta_labels, parsed_elems))


def crawl_result_sites():
    scraper = RaceResultScaraper()
    base_url = "https://keiba.yahoo.co.jp/race/result/"

    for year in range(20, 21):
        for loc in range(5, 11):
            for kai in range(1, 6):
                for nichi in range(1, 9):
                    for round in range(1, 13):
                        date_url = "{:02}{:02}{:02}{:02}{:02}".format(year, loc, kai, nichi, round)
                        fetch_url = base_url + date_url
                        json_file = "results/{}.json".format(date_url)
                        json_file = Path(json_file).resolve()

                        print("[fetch] URL: %s ..." % fetch_url)
                        result_dict = fetch_result_site(scraper, fetch_url)
                        if result_dict is None:
                            continue

                        print("[dump] URL: %s ..." % fetch_url)
                        dump_dict_as_json(json_file, result_dict)
                        time.sleep(1)
    print("[crawl] all done")


def fetch_result_site(scraper, url):
    try:
        soup = scraper.retrieve_html(url)
    except requests.exceptions.HTTPError as e:
        print("[fetch] ERROR: %s ..." % e)
        return None

    # todo: なぜか"https://keiba.yahoo.co.jp/schedule/list/"に飛ぶURLを暫定処置で排除
    try:
        score_dicts = scraper.extract_scores(soup)
    except IndexError as e:
        print("[fetch] ERROR: %s ..." % e)
        return None
    scraper.parse_scores(score_dicts)

    racehead_dict = scraper.extract_racehead(soup)
    scraper.parse_racehead(racehead_dict)

    result_dict = {
        "scores": score_dicts,
        "racehead": racehead_dict,
    }

    return result_dict


def dump_dict_as_json(json_file, data_dict):
    with open(json_file, "w") as f:
        json.dump(data_dict, f, indent=4)


def test_scores(url):
    scraper = RaceResultScaraper()
    soup = scraper.retrieve_html(url)
    score_dicts = scraper.extract_scores(soup)
    scraper.parse_scores(score_dicts)
    return score_dicts


def test_racehead(url):
    scraper = RaceResultScaraper()
    soup = scraper.retrieve_html(url)
    racehead_dict = scraper.extract_racehead(soup)
    scraper.parse_racehead(racehead_dict)
    return racehead_dict


def test_result_site():
    url = "https://keiba.yahoo.co.jp/race/result/2003010701/"
    score_dicts = test_scores(url)
    pprint(score_dicts)
    racehead_dict = test_racehead(url)
    pprint(racehead_dict)


def parse_week(data):
    if data == '日':
        return 0
    elif data == '月':
        return 1
    elif data == '火':
        return 2
    elif data == '水':
        return 3
    elif data == '木':
        return 4
    elif data == '金':
        return 5
    elif data == '土':
        return 6


# def parse_start_time(data):
#     base_time = datetime.strptime("00:00", "%H:%M")
#     time = datetime.strptime(v, "%H:%M")
#     min = abs(time - base_time).total_seconds / 60.0  # fixme: timedeltaをfloatでは割れない
#     return min


def parse_weather(data):
    if data == '晴':
        return 0
    elif data == '曇':
        return 1
    elif data == '小雨':
        return 2
    elif data == '雨':
        return 3
    elif data == '雪':
        return 4


def parse_condition(data):
    if data == '良':
        return 0
    elif data == '稍重':
        return 1
    elif data == '重':
        return 2
    elif data == '不良':
        return 3


def parse_horse_sex(data):
    if data == '牡':
        return 0
    elif data == '牝':
        return 1
    elif data == 'せん':
        return 2


def parse_horse_weight_diff(data):
    if data == ' - ':
        return 0
    else:
        return data


def parse_jockey_weight(data):
    if isinstance(data, str):
        data = re.sub(re.compile("[☆△▲★◇]"), "", data)
        return float(data)
    elif isinstance(data, float):
        return data


def parse_arrival_order(data):
    data = str(data)
    s = re.search(re.compile(r"(\d+)(\(\d+\))*"), data)
    if s:
        return int(s.group(1))
    else:
        return data


def load_race_data():
    list_score_and_racehead = []
    list_arrival_order = []
    list_time = []

    current_dir = Path.cwd()
    results_fir = current_dir / 'results'
    json_files = results_fir.glob('*.json')

    # jsonファイルを全読み込み
    for json_file in json_files:
        data_dict = load_json_as_dict(json_file)
        score_and_racehead, arrival_orders, times = parse_loaded_data(data_dict)
        list_score_and_racehead.append(score_and_racehead)
        list_arrival_order.append(arrival_orders)
        list_time.append(times)
        # todo: delete
        print("{}: {}".format(json_file, str(len(score_and_racehead))), end="")
        for sr in score_and_racehead:
            print(len(sr))

    # pprint(list_score_and_racehead)
    # pprint(list_arrival_order)
    return list_score_and_racehead, list_arrival_order, list_time


# def load_time():
#     list_time = []
#
#     current_dir = Path.cwd()
#     results_fir = current_dir / 'results'
#     json_files = results_fir.glob('*.json')
#
#
#     return list_time


def parse_loaded_data(data_dict):
    effective_score_labels = [
        "arrival_order",
        "frame_no",
        "horse_no",
        # "horse_name",
        "horse_sex",
        "horse_age",
        "horse_weight",
        "horse_weight_diff",
        # "horse_b",
        # "arrival_diff",
        "time",
        # "last_3f_time",
        # "passing_order_1st",
        # "passing_order_2nd",
        # "passing_order_3rd",
        # "passing_order_4th",
        # "jockey_name",
        "jockey_weight",
        "odds",
        "popularity",
        # "trainer_name",
    ]

    effective_racehead_labels = [
        "race_no",
        # "date",
        "week",
        "kai",
        # "lacation",
        "nichi",
        # "start_time",
        "weather",
        "condition",
    ]

    # ネスト化された辞書を平坦化
    score_dicts = []
    racehead_dict = {}
    for k, v in data_dict.items():
        if k == 'scores':
            for s in v:
                flattened_dict = flatten_dict(s)
                score_dict = {k_: v_ for k_, v_ in flattened_dict.items() if k_ in effective_score_labels}
                score_dicts.append(score_dict)
            score_dicts.sort(key=lambda x: x['horse_no'])
        elif k == 'racehead':
            flattened_dict = flatten_dict(v)
            racehead_dict = {k_: v_ for k_, v_ in flattened_dict.items() if k_ in effective_racehead_labels}

    # データを整形
    score_and_racehead = []

    # racehead
    racehead = []
    for k, v in racehead_dict.items():
        if k == 'week':
            racehead.append(parse_week(v))
        # elif k == 'start_time':
        #     racehead.append(parse_start_time(v))
        elif k == 'weather':
            racehead.append(parse_weather(v))
            if parse_weather(v) is None:
                print(v)
                raise
        elif k == 'condition':
            racehead.append(parse_condition(v))
        else:
            racehead.append(v)

    arrival_orders = []
    times = []
    for score_dict in score_dicts:
        # score
        score = []
        for k, v in score_dict.items():
            if k == 'arrival_order':
                arrival_orders.append(parse_arrival_order(v))
            elif k == 'time':
                times.append(v)
            elif k == 'horse_sex':
                score.append(parse_horse_sex(v))
            elif k == 'horse_weight_diff':
                score.append(parse_horse_weight_diff(v))
            elif k == 'jockey_weight':
                score.append(parse_jockey_weight(v))
            else:
                score.append(v)
        score_and_racehead.append(score + racehead)
    return score_and_racehead, arrival_orders, times


def fetch_predicting_data(url):
    scaraper = RaceResultScaraper()
    result_dict = fetch_result_site(scaraper, url)

    score_and_racehead, _, _ = parse_loaded_data(result_dict)
    return score_and_racehead


### TEST CODE ###
def test_RaceResultScraper_extract_scores():
    url = 'https://keiba.yahoo.co.jp/race/result/2005040811/'
    c = Crawler()
    soup = c.fetch_url(url)

    s = RaceResultScaraper()
    d = s.extract_scores(soup)
    s.parse_scores(d)
    pprint(d)


def test_RaceResultScraper_extract_racehead():
    urls = [
        'https://keiba.yahoo.co.jp/race/result/2005040811/',
        'https://keiba.yahoo.co.jp/race/result/1908050505/',
    ]
    c = Crawler()

    for url in urls:
        soup = c.fetch_url(url)
        s = RaceResultScaraper(soup)
        d = s.extract_racehead()
        s.parse_racehead(d)
        pprint(d)

def test_DirectoryHorseScraper_extract():
    urls = [
        'https://keiba.yahoo.co.jp/directory/horse/2015104961/',
        'https://keiba.yahoo.co.jp/directory/horse/2015104688/',
        'https://keiba.yahoo.co.jp/directory/horse/2011102151/',
    ]
    c = Crawler()

    for url in urls:
        soup = c.fetch_url(url)
        s = DirectoryHorseScraper(soup)
        d = s.extract()
        s.parse(d)
        pprint(d)
### TEST CODE ###


if __name__ == "__main__":
    test_RaceResultScraper_extract_racehead()
