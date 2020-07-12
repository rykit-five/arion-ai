import sqlite3
from sqlite3 import Error

import scraper


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
                                race_id INTEGER,
                                arrival_order INTEGER NOT NULL,
                                frame_no INTEGER NOT NULL,
                                horse_no INTEGER NOT NULL, 
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
                                trainer_name TEXT NOT NULL,
                                PRIMARY KEY (race_id, horse_name)
                             )''')
        self.conn.commit()

    def close(self):
        self.curs.close()
        self.conn.close()

    def insert(self, score):
        sql = '''INSERT INTO scores (
                    arrival_order,
                    frame_no,
                    horse_no,
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

    def select(self):
        pass


def test_register(score_dicts):
    db_file = "test_db.sqlite"
    reg = Register(db_file)
    reg.create(reset=False)
    for score_dict in score_dicts:
        reg.insert(tuple(score_dict.values()))


if __name__ == '__main__':
    url = "https://keiba.yahoo.co.jp/race/result/1705050211"
    score_dicts = scraper.test_scraper(url)
    test_register(score_dicts)