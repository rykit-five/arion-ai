#!/usr/bin/python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
from pprint import pprint


class Crawler(object):

    def __init__(self):
        pass

    def crawl(self, base_url, url_rule):
        """
        base_urlを起点にurl_ruleにマッチするサイトをクロール
        :param base_url: str
        :param url_rule: re.compile obj
        """
        pass

    def fetch_url(self, url):
        """
        urlからhtml(soup)を取得
        :param url: str
        :return: BeautifulSoup obj: html
        """
        req = requests.get(url)
        req.raise_for_status()
        soup = BeautifulSoup(req.content, "html.parser")
        return soup

    def get_urls(self, soup, url_rule):
        """
        html(soup)からurl_ruleにマッチするurlを取得
        :param soup: BeautifulSoup obj: html
        :param url_rule: re.compile obj
        :return: urls: list
        """
        urls = []
        return urls


def test_crawler_fetch_url():
    """
    Crawlerクラス fetch_urlメソッドのテスト
    """
    url = 'https://keiba.yahoo.co.jp/race/result/2005040811/'
    crawler = Crawler()
    soup = crawler.fetch_url(url)
    print(soup)


if __name__ == '__main__':
    test_crawler_fetch_url()