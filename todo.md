# TODO

## Component
### Class
* Crawler ... 
  + 巡回ルールに従って巡回するメソッドを作成

* Scraper
  + 抽象クラス化してインタフェースをまとめる
  
* RaceResultScraper
  + レースの基本情報を抽出するメソッドを整形
  + スコアのhorse_infoのパースを修正              ... OK
    * str to int / floatのキャスト                ... OK
    * 結果を格納する辞書の一次元化                ... OK
  + スコアのpassing_orderのパースを修正           ... OK
    * str to int / floatのキャスト                ... OK
    * 結果を格納する辞書の一次元化                ... OK
  
* DirectoryHorseScraper
  + 巡回URLを抽出するメソッドを追加(以下も同じ)
  + 馬の基本情報を抽出するメソッドを追加
  
* DirectoryJockieScraper
  + 騎手の基本情報を抽出するメソッドを追加
  
* DirectoryTrainerScraper
  + 厩舎の基本情報を抽出するメソッドを追加
  
* Register
  + テーブルを作成するメソッドを作成
  + テーブルを初期化するメソッドを作成
  + テーブルにレコードを挿入するメソッドを作成
  + テーブルからレコードを取得するメソッドを作成
  + 取得したレコードからデータを集計するメソッドを作成
  
* Util
  + すべてのメソッドをstatic化                    ... OK


