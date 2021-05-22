import pandas as pd
import numpy as np
import requests, bs4
from raceurl import *

def str2num(df):
    series = df.copy()

    if series.apply(lambda s:pd.to_numeric(s, errors='coerce')).notnull().all() == False:
        series[series.apply(lambda s:pd.to_numeric(s, errors='coerce')).notnull()== False] = 16
    
    return series

def scraping(url):

    # 表内の馬名とそのリンクを取得
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.content, 'html.parser')
    horse = soup.find_all('span', class_='Horse_Name')
    jockey = soup.find_all('td', class_='Jockey')

    title = soup.find('title').text
    print(url)
    print(title)

    horse_win_rates = []
    horse_median = []
    horse_popularity_median = []

    jockey_win_rates = []
    jockey_median = []

    # 馬の名前と馬情報が載っているリンクを取得
    for h in horse:
        table = pd.read_html(h.find('a').get('href'))

        history = None

        for i in range(len(table)):
            if '着順' in table[i].columns:
                history = table[i]
                break

        places = str2num(history['着順'])
        places = places.to_numpy().astype(int)

        first = places[places==1]
        median = np.median(places)

        # 人気集計
        popularity = str2num(history['人気'])
        popularity = popularity.to_numpy().astype(int)
        p_median = np.median(popularity)

        horse_win_rates.append(len(first)/len(places)) # 1着の試合 / 全試合
        horse_median.append(median)
        horse_popularity_median.append(p_median)

    # ジョッキー情報
    for i in range(len(horse)):
        table = pd.read_html(jockey[i].find('a').get('href'))
        history = table[0]

        places = str2num(history['着順'])
        places = places.to_numpy().astype(int)

        first = places[places==1]
        median = np.median(places)

        jockey_win_rates.append(len(first)/len(places))
        jockey_median.append(median)

    # 表全体をDataFrameで取得
    table = pd.read_html(url)
    result = table[0]
    drop_col = ['タイム', '着差', 'コーナー通過順', '厩舎', '騎手', '馬名']
    result = result.drop(drop_col, axis=1)

    # 性齢を分割
    y_s = result['性齢'].values.tolist()

    sex = [y_s[i][0:1] for i in range(len(y_s))]
    y_old = [y_s[i][-1] for i in range(len(y_s))]

    sex_np = np.zeros([len(sex), 2]).astype(int)

    # 性別振り分け
    for i, s in enumerate(sex):
        if s == '牡': sex_np[i][0] = 1
        elif s == '牝': sex_np[i][1] = 1
    
    # 性別用のデータフレーム
    df_sex = pd.DataFrame(sex_np, columns=['牡', '牝'])

    # ooo(±o)で抽出しているので，分割
    weight_pm = result['馬体重(増減)'].values.tolist()
    weight, pm = [], []

    for wp in weight_pm:
        try:
            w, p = wp.split('(')
            p, _ = p.split(')') 
        except:
            w = wp
            p = 0

        weight.append(w)
        pm.append(p)

    # 馬体重(増減)のカラムを分割
    result = result.drop('馬体重(増減)', axis=1)
    result = result.drop('性齢', axis=1)
    result['年齢'] = y_old
    result['馬体重'] = pd.Series(weight)
    result['体重増減'] = pd.Series(pm)
    result['人気中央値'] = horse_popularity_median
    result['馬勝率'] = pd.Series(horse_win_rates)
    result['馬着中央値'] = horse_median
    result['騎手勝率'] = jockey_win_rates
    result['騎手着中央値'] = jockey_median

    result = pd.concat([result, df_sex], axis=1)

    # 取消や除外となっている行以外を抽出
    if result['着順'].apply(lambda s:pd.to_numeric(s, errors='coerce')).notnull().all() == False:
        result = result[result['着順'].apply(lambda s:pd.to_numeric(s, errors='coerce')).notnull()!=False]

    # 人気順(昇順)にソート
    result = result.sort_values('人気')

    _, id = url.split('id=')
    id, _ = id.split('&rf')

    # csvで出力
    result.to_csv(f'./dataset/{id}.csv', encoding='shift-jis', index=False)

if __name__ == '__main__':

    for url in urls_2015:
        scraping(url)

    for url in urls_2016:
        scraping(url)

    for url in urls_2017:
        scraping(url)

    for url in urls_2018:
        scraping(url)

    for url in urls_2019:
        scraping(url)

    for url in urls_2020:
        scraping(url)

    for url in urls_2021:
        scraping(url)