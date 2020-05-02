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
                        dump_site_as_json(json_file, result_dict)
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


def dump_site_as_json(json_file, data_dict):
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


if __name__ == "__main__":
    crawl_result_sites()
    # test_result_site()
