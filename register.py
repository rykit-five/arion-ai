import sqlite3
import logging
from pprint import pprint

import scraper

### FOT TEST ###
from scraper import test_RaceResultScraper_extract_racehead, test_RaceResultScraper_extract_scores
### FOT TEST ###

class Register_old:

    #    sql_create_score = '''CREATE TABLE IF NOT EXISTS scores (
    #                             race_id INTEGER,
    #                             arrival_order INTEGER NOT NULL,
    #                             frame_no INTEGER NOT NULL,
    #                             horse_no INTEGER NOT NULL,
    #                             horse_name TEXT NOT NULL,
    #                             horse_info TEXT NOT NULL,
    #                             arrival_diff TEXT,
    #                             time REAL NOT NULL,
    #                             last3f_time REAL NOT NULL,
    #                             passing_order TEXT NOT NULL,
    #                             jockey_name TEXT NOT NULL,
    #                             jockey_weight REAL NOT NULL,
    #                             odds REAL NOT NULL,
    #                             popularity REAL NOT NULL,
    #                             trainer_name TEXT NOT NULL,
    #                             PRIMARY KEY (race_id, horse_name)
    #                             )'''
    # #
    # sql_create_xxx = ''''''
    # spl_delete_score = '''DROP TABLE IF EXISTS scores'''
    # sql_insert_xxx = ''''''
    # sql_select_xxx = ''''''


    def __init__(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)
        self.conn = conn
        self.curs = self.conn.cursor()

    def create_table(self, trg_table=""):
        # sql_create = eval("self.sql_create_{}".format(trg_table))
        sql_create = self.__dict__["self.sql_create_{}".format(trg_table)]

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

    def delete_table(self, table="all"):
        self.curs.execute('''DROP TABLE IF EXISTS scores''')



    def close_connect(self):
        self.curs.close()
        self.conn.close()

    def insert(self, record, trg_table=""):
        if trg_table == "condition":
            self.curs.execute(sql, record)
            pass
        elif trg_table == "race":
            pass
        elif trg_table == "score":
            pass
        elif trg_table == "horse":
            pass
        elif trg_table == "jockey":
            pass
        elif trg_table == "trainer":
            pass
        else:
            raise

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


logger = logging.getLogger(__name__)


class Register(object):
    PRIMARY_KEY = ('id', int)

    def __init__(self, filename, debug=True, smart_pdate=False):
        self.db = sqlite3.connect(filename, check_same_thread=False)
        self.debug = debug
        # self.smart_update = smart_pdate
        self.registered = {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _execute(self, sql, args=None):
        if args is None:
            if self.debug:
                logger.debug('%s', sql)
            return self.db.execute(sql)
        else:
            if self.debug:
                logger.debug('%s, %r', sql, args)
            return self.db.execute(sql, args)

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.isolation_level = None
        self._execute('VACUUM')
        self.db.close()

    def register(self, table, slots):
        if table in self.registered.keys():
            raise TypeError('%s is already registered' % table)
        self.registered[table] = slots

    def _schema(self, table):
        if table not in self.registered.keys():
            raise TypeError('%s was never registered' % table)
        return self.registered[table]

    def get_slots(self, table_name):
        sql = 'PRAGMA table_info ({})'.format(table_name)
        cur = self._execute(sql)
        rows = cur.fetchall()
        if len(rows) != 0:
            return [(r[1], r[2]) for r in rows]
        else:
            return []

    def create_table(self, table_name, slots):
        """
        tableがslotsと同じカラム名と型を持つか調査
         * カラム名が同じで型が異なる場合 -> 例外
         * カラムが足りない場合           -> テーブルに追加
         * そもそもテーブルがない場合     -> 作成
        :param table: str
        :param slots: list of tuple (str: name, str: type_)
        :return: None or TypeError
        """
        available = self.get_slots(table_name)

        def column(name, type_, primary=True):
            if (name, type_) == self.PRIMARY_KEY and primary:
                return 'INTEGER PRIMARY KEY'
            elif type_ in (int, bool):
                return 'INTEGER'
            elif type_ in (float, ):
                return 'REAL'
            elif type_ in (bytes, ):
                return 'BLOB'
            else:
                return 'TEXT'

        if len(available) != 0:
            print(available)
            # modified_slots = [(name, type_) for name, type_ in slots
            #                   if name in (name for name, _ in available)
            #                   and (name, column(name, type_, primary=False)) not in available]
            # for name, type_ in modified_slots:
            #     raise TypeError('Column {} is {}, but expected {}'.format(
            #         name, next(dbtype for n, dbtype in available if n == name), column(name, type_)))
            # missing_slots = [(name, type_) for name, type_ in slots if name not in (n for n, _ in available)]
            # for name, type_ in missing_slots:
            #     self._execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, name, column(name, type_)))
            pass
        else:
            sql = 'CREATE TABLE %s (%s)' % (
                table_name, ', '.join('{} {}'.format(n, column(n, t)) for (n, t) in slots))
            self._execute(sql)
            self.register(table_name, slots)


    def _update(self, o):
        # update
        pass

    def insert(self, table_name, args):
        """
        tableからカラム名(name)を取得してレコード(args)を挿入
        :param table: str
        :param args: dict
        :return: lastrowid
        """
        slots = self.get_slots(table_name)
        slots = [(n, t) for n, t in slots if (n, t) != self.PRIMARY_KEY]

        # check columns
        slot_names = [n for (n, t) in slots]
        arg_names = args.keys()
        # if len(slot_names) != len(arg_names):
            # raise KeyError("Args has key '{}' not defined in the table".format(', '.join(set(slot_names) - set(arg_names))))

        arg_values = [args[sn] for sn in slot_names]
        str_names = ', '.join([sn for sn in slot_names])
        # str_values = ', '.join([av for av in arg_values])
        sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, str_names, ', '.join('?' * len(slot_names)))
        return self._execute(sql, arg_values).lastrowid

    def delete_table(self, table_name):
        sql = 'DROP TABLE IF EXISTS {}'.format(table_name)
        return self._execute(sql)

    def query(self, o, select=None, where=None, order_by=None, group_by=None, limit=None):
        sql = []
        args = []


        pass

    def load(self, o, *args, **kwargs):
        # insert
        pass

    def get(self, o, *args, **kwargs):
        # select
        pass


# def test_register():
#     db_file = "test_db.sqlite"
#     reg = Register(db_file)
#     reg.create(reset=False)
#     for score_dict in score_dicts:
#         reg.insert(tuple(score_dict.values()))


### TEST CODE ###
SLOT_CONDITION = [
    #("id", int),
    ("title", str),
    ("date", str),
    ("week", str),
    ("month", int),
    ("day", int),
    ("round", int),
    ("start_time", str),
    ("weather", str),
    ("ground", str),
]

SLOT_RACE = [
    ("title", str),
    ("grade", str),
    ("age", str),
    ("class", str),
    ("reward_1st", int),
    ("reward_2nd", int),
    ("reward_3rd", int),
    ("reward_4th", int),
    ("reward_5th", int),
    ("location", str),
    ("surface", str),
    ("clockwise", str),
    ("distance", int),
]

SLOT_SCORE = [
    ("arrival_order", int),
    ("frame_no", int),
    ("horse_no", int),
    ("horse_name", str),
    ("horse_sex", str),
    ("horse_age", int),
    ("horse_weight", float),
    ("horse_weight_diff", float),
    ("horse_blinker", str),
    ("time", float),
    ("arrival_diff", str),
    ("passing_order_1st", int),
    ("passing_order_2nd", int),
    ("passing_order_3rd", int),
    ("passing_order_4th", int),
    ("last_3f_time", float),
    ("jockey_name", str),
    ("jockey_weight", float),
    ("popularity", int),
    ("odds", float),
    ("trainer_name", str),
]


def test_create_table():
    reg = Register("test_db.sqlite", debug=True)
    reg.create_table("CONDITION", SLOT_CONDITION)
    reg.create_table("RACE", SLOT_RACE)
    reg.create_table("SCORE", SLOT_SCORE)
    reg.commit()


def test_insert():
    reg = Register("test_db.sqlite", debug=True)
    test_insert_condition(reg)
    test_insert_race(reg)
    test_insert_scores(reg)


def test_insert_condition(reg):
    args_list = test_RaceResultScraper_extract_racehead()
    for args in args_list:
        # args_condition = [v for k, v in args.items() if k in [name for name, tyep_ in SLOTS_CONDITION]]
        reg.insert("CONDITION", args)
        reg.commit()


def test_insert_race(reg):
    args_list = test_RaceResultScraper_extract_racehead()
    for args in args_list:
        reg.insert("RACE", args)
        reg.commit()


def test_insert_scores(reg):
    args_list = test_RaceResultScraper_extract_scores()
    for args in args_list:
        for a in args:
            reg.insert("SCORE", a)
            reg.commit()


def test_delete():
    reg = Register("test_db.sqlite", debug=True)
    reg.delete_table("RACE")
    reg.delete_table("CONDITION")

### TEST CODE ###


if __name__ == '__main__':
    # logging.basicConfig(filename='logfile/logger.log', level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    test_create_table()
    test_insert()
    # test_delete()

