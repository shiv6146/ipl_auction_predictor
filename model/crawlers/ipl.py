'''
Basic crawler to scrape current IPL men's team list and auction player stats from iplt20.com
'''
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from .cricbuzz import Player

class Team:
    def __init__(self) -> None:
        pass

def get_current_teams():
    r = requests.get('https://www.iplt20.com/teams/men')
    if r.status_code != 200:
        return []
    bs = BeautifulSoup(r.content, 'lxml')
    return [i.get_text().lower().replace(' ', '-') for i in bs.select('h3')]

def get_page_content(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.content

def get_sold_players():
    start_year = 2013
    current_year = datetime.now().year
    current_teams = get_current_teams()
    d = {
        'player': [],
        'role': [],
        'team': [],
        'year': [],
        'price': []
    }
    id_num = 35
    while start_year <= current_year:
        # sold players selector without team name
        pattern = f'#autab{id_num} :nth-child(1) tbody' if start_year < 2022 else '.ih-pt-tab-bg :nth-child(1) tbody'
        content = get_page_content(f'https://www.iplt20.com/auction/{start_year}')
        if content is None:
            start_year += 1
            continue
        bs = BeautifulSoup(content, 'lxml')
        players_list = bs.select(pattern)
        # table DOM structure have changed since 2022 in iplt20.com
        if len(players_list) > 0 and start_year >= 2022:
            players_list = players_list[3:len(players_list)-1]
        for player_list in players_list:
            team = player_list.fetchPrevious(name='h2', limit=1)
            if len(team) == 0:
                continue
            team_name = team[0].text.lower().strip().replace(' ', '-')
            # handle changed team names and ignore teams that have been removed
            if team_name == 'kings-xi-punjab':
                team_name = 'punjab-kings'
            elif team_name == 'deccan-chargers':
                team_name = 'sunrisers-hyderabad'
            elif team_name == 'delhi-daredevils':
                team_name = 'delhi-capitals'
            elif team_name not in current_teams:
                continue
            players_per_team = player_list.find_all('td')
            # older auction tables had only 3 columns and the newer ones added a new column
            if start_year < 2022:
                for i in range(0, len(players_per_team), 3):
                    d['player'].append(players_per_team[i].text.strip())
                    d['role'].append(players_per_team[i+1].text.strip())
                    d['team'].append(team_name)
                    d['year'].append(start_year)
                    price = players_per_team[i+2].text.strip()
                    # 2013 usd to inr exchange rate 58
                    if price.startswith('$'):
                        price = int(price[1:].replace(',', '')) * 58
                    else:
                        price = int(price[1:].replace(',', ''))
                    d['price'].append(price)
            else:
                for i in range(0, len(players_per_team), 4):
                    d['player'].append(players_per_team[i].text.strip())
                    d['role'].append(players_per_team[i+2].text.strip())
                    d['team'].append(team_name)
                    d['year'].append(start_year)
                    price = players_per_team[i+3].text.strip()
                    # 2013 usd to inr exchange rate 58
                    if price.startswith('$'):
                        price = int(price[1:].replace(',', '')) * 58
                    else:
                        price = int(price[1:].replace(',', ''))
                    d['price'].append(price)
        start_year += 1
        id_num -= 4
    return pd.DataFrame(d)

def extract_player_feature_vector(player: Player) -> list:
    # feature_names = [
    #     "name", "country", "age", "height", "role", "bat_style", "bowl_style", "t20_no", "t20_runs", "t20_avg", "t20_sr", "t20_50", "t20_4s", "t20_6s",
    #     "ipl_no", "ipl_runs", "ipl_avg", "ipl_sr", "ipl_50", "ipl_4s", "ipl_6s", "t20_wkts", "t20_bowl_econ", "t20_bowl_avg", "t20_bowl_sr",
    #     "ipl_wkts", "ipl_bowl_econ", "ipl_bowl_avg", "ipl_bowl_sr", "team", "year", "price"
    # ]
    features = [
        player.name,
        player.country,
    ]
    # process player info
    for k,v in player.info.items():
        if k == 'age':
            features.append(int(v))
        elif k != 'age' and k != 'height':
            features.append(v.replace(' ', '-'))
        else:
            features.append(v)
    # process player bat stats
    bat_stat_cols = ['no', 'runs', 'avg', 'sr', '50', '4s', '6s']
    for bat_stat in player.bat_stats.loc['t20i'][bat_stat_cols]:
        features.append(bat_stat)
    for bat_stat in player.bat_stats.loc['ipl'][bat_stat_cols]:
        features.append(bat_stat)
    # process player bowl stats
    bowl_stat_cols = ['wkts', 'econ', 'avg', 'sr']
    for bowl_stat in player.bowl_stats.loc['t20i'][bowl_stat_cols]:
        features.append(bowl_stat)
    for bowl_stat in player.bowl_stats.loc['ipl'][bowl_stat_cols]:
        features.append(bowl_stat)
    return features

def build_dataset():
    player_cache = {}
    # feature vector column names
    feature_names = [
        "name", "country", "age", "height", "role", "bat_style", "bowl_style", "t20_no", "t20_runs", "t20_avg", "t20_sr", "t20_50", "t20_4s", "t20_6s",
        "ipl_no", "ipl_runs", "ipl_avg", "ipl_sr", "ipl_50", "ipl_4s", "ipl_6s", "t20_wkts", "t20_bowl_econ", "t20_bowl_avg", "t20_bowl_sr",
        "ipl_wkts", "ipl_bowl_econ", "ipl_bowl_avg", "ipl_bowl_sr", "team", "year", "price"
    ]
    data = []
    auction_fname = 'auction_data.csv'
    data_fname = 'data.csv'
    if not os.path.isfile(auction_fname):
        get_sold_players().to_csv(auction_fname, index=False)
    auction_df = pd.read_csv(auction_fname)
    for i in auction_df.iterrows():
        p = None
        player_feat_vec = []
        if i[1].player in player_cache:
            player_feat_vec = list(player_cache[i[1].player])
            player_feat_vec.append(i[1].team)
            player_feat_vec.append(i[1].year)
            player_feat_vec.append(i[1].price)
            if len(player_feat_vec) != len(feature_names):
                print(f'[Cache] Strange mismatch: {i[1].player}, {player_feat_vec}, {player_cache[i[1].player]}')
                exit()
            data.append(player_feat_vec)
            print(f'Processed {i[0]} / {auction_df.shape[0]}: Player - {i[1].player}')
            continue
        try:
            p = Player(i[1].player)
            # some players are better searched using their lastnames
            if p.bat_stats is None or p.bowl_stats is None:
                last_name = i[1].player.split()[-1]
                if last_name != i[1].player:
                    p_tmp = Player(last_name)
                    first_name = p_tmp.name.split()[0]
                    if p.name.startswith(first_name):
                        p = p_tmp
        except Exception as e:
            print(f'Unable to fetch player stats: {i[1].player}')
        if p is None:
            print(f'Failed to construct player profile: {i[1].player}')
            continue
        # skipping players with empty stats
        if p.bowl_stats is None or p.bat_stats is None:
            print(f'Skipping player with empty stats: {i[1].player}')
            continue
        # append each player feature vector
        player_feat_vec = extract_player_feature_vector(p)
        # add to player_cache only the player_feat_vec without auction data
        player_cache[i[1].player] = list(player_feat_vec)
        player_feat_vec.append(i[1].team)
        player_feat_vec.append(i[1].year)
        player_feat_vec.append(i[1].price)
        if len(player_feat_vec) != len(feature_names):
            print(f'Strange mismatch: {i[1].player}, {player_feat_vec}, {player_cache[i[1].player]}')
            exit()
        data.append(player_feat_vec)
        print(f'Processed {i[0]} / {auction_df.shape[0]}: Player - {i[1].player}')
        time.sleep(1)
    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv(data_fname, index=False)