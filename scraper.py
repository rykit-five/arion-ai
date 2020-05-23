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

from utils import load_json_as_dict, flatten_dict


class Scraper(object):

    def __init__(self):
        pass

    def retrieve_html(self, url):
        req = requests.get(url)
        req.raise_for_status()
        soup = BeautifulSoup(req.content, "html.parser")
        return soup


class ResultScaraper(Scraper):
    racehead_labels = (
        "race_no",
        "tit",
        "title",
        "meta",
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
        racehead_dict = dict(zip(self.racehead_labels, racehead))
        return racehead_dict

    def parse_racehead(self, racehead_dict):
        racehead_dict["race_no"] = self._parse_race_no(racehead_dict["race_no"])
        racehead_dict["tit"] = self._parse_tit(racehead_dict["tit"])
        # racehead_dict["meta"] = self.parse_meta(recehead_dicts["meta"])

        for k, v in racehead_dict.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    v[_k] = self._str_to_digit(_v)
            else:
                racehead_dict[k] = self._str_to_digit(v)

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
                span_tags = td_tag.select("span.scdItem")
                if span_tags:
                    score.append(span_tags[0].get_text().strip("()\t\n\x0b\x0c\r "))
                    span_tags[0].extract()
                # 上記以外
                td_text = td_tag.get_text().strip()
                if td_text:
                    score.append(td_text)
            # データを保存
            if score[0] in ["中止", "除外", "取消"]:
                continue
            assert len(self.score_labels) == len(score)
            score_dict = dict(zip(self.score_labels, score))
            score_dicts.append(score_dict)
        return score_dicts

    def parse_scores(self, score_dicts):
        for score_dict in score_dicts:
            score_dict["time"] = self._to_sec(score_dict["time"])
            # score_dict["last3f_time"] = self._to_sec(score_dict["last3f_time"])
            score_dict["horse_info"] = self._parse_horse_info(score_dict["horse_info"])
            score_dict["passing_order"] = self._parse_passing_order(score_dict["passing_order"])

            for k, v in score_dict.items():
                if isinstance(v, dict):
                    for _k, _v in v.items():
                        v[_k] = self._str_to_digit(_v)
                else:
                    score_dict[k] = self._str_to_digit(v)

    def _parse_horse_info(self, horse_info):
        s = re.search(re.compile("(牡|牝|せん)(\d+)/(\d+)\(([\+\-]?\d+| \- )\)/(B?)"), horse_info)
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
        return re.sub(r"R", "", data)

    def _parse_tit(self, data):
        d = data.split("|")

        # todo: 例外処理
        if len(d) != 3:
            raise

        parsed = []
        s = re.search(re.compile("(\d{4}年\d{1,2}月\d{1,2}日)（(日|月|火|水|木|金|土)）"), d[0])
        if s:
            parsed.extend(s.groups())
        s = re.search(re.compile("(\d{1,2})回(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)(\d{1,2})日"), d[1])
        if s:
            parsed.extend(s.groups())
        s = re.search(re.compile("(\d{1,2}:\d{1,2})発走"), d[2])
        if s:
            parsed.extend(s.groups())
        return dict(zip(self.tit_labels, parsed))

    def _pareta(self, meta):
        pass

    def _parse_title(self, title):
        pass

def crawl_result_sites():
    scraper = ResultScaraper()
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
    scraper = ResultScaraper()
    soup = scraper.retrieve_html(url)
    score_dicts = scraper.extract_scores(soup)
    scraper.parse_scores(score_dicts)
    return score_dicts


def test_racehead(url):
    scraper = ResultScaraper()
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
    elif data == '雨':
        return 2
    elif data == '雪':
        return 3


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
    s = re.search(re.compile('(\d{1,2})\(\d{1,2}\)'))
    if s:
        return float(s.group(0))
    else:
        return data


def load_race_data():
    list_score_and_racehead = []
    list_arrival_order = []

    current_dir = Path.cwd()
    results_fir = current_dir / 'results'
    json_files = results_fir.glob('200301*.json')

    # jsonファイルを全読み込み
    for json_file in json_files:
        data_dict = load_json_as_dict(json_file)
        score_and_racehead, arrival_orders = parse_loaded_data(data_dict)
        list_score_and_racehead.append(score_and_racehead)
        list_arrival_order.append(arrival_orders)
        # todo: delete
        print("{}: {}".format(json_file, str(len(score_and_racehead))), end="")
        for sr in score_and_racehead:
            print(len(sr))

    # pprint(list_score_and_racehead)
    # pprint(list_arrival_order)
    return list_score_and_racehead, list_arrival_order


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
        "last3f_time",
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
        elif k == 'condition':
            racehead.append(parse_condition(v))
        else:
            racehead.append(v)

    arrival_orders = []
    for score_dict in score_dicts:
        # score
        score = []
        for k, v in score_dict.items():
            if k == 'arrival_order':
                arrival_orders.append(v)
            elif k == 'horse_sex':
                score.append(parse_horse_sex(v))
            elif k == 'horse_weight_diff':
                score.append(parse_horse_weight_diff(v))
            elif k == 'jockey_weight':
                score.append(parse_jockey_weight(v))
            else:
                score.append(v)
        score_and_racehead.append(score + racehead)
    return score_and_racehead, arrival_orders


def fetch_predicting_data(url):
    scaraper = ResultScaraper()
    result_dict = fetch_result_site(scaraper, url)

    score_and_racehead, _ = parse_loaded_data(result_dict)
    return score_and_racehead


if __name__ == "__main__":
    crawl_result_sites()
    # test_result_site()
