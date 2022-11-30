'''
Basic crawler to search a player using name and get player's T20I and IPL stats along with player info from cricbuzz
'''
import requests
import pandas as pd
from bs4 import BeautifulSoup

class Player:
    def __init__(self, name=None):
        self._bs = None
        self.id = None
        self.name = name
        self.country = None
        self.info = None
        self.bat_stats = None
        self.bowl_stats = None
        if name is not None:
            p = self.get(name)
            if p is not None:
                self.id = p['id']
                self.name = p['title']
                self.country = p['country']
                self._update_soup(self.id)
                self.info = self.get_info()
                self.bat_stats, self.bowl_stats = self.get_stats()
    
    def __str__(self):
        return f'''
        Name: {self.name}
        Country: {self.country}
        Info: {self.info}
        Batting Stats: {self.bat_stats}
        Bowling Stats: {self.bowl_stats}
        '''
    
    def _update_soup(self, id):
        url = f'https://www.cricbuzz.com/profiles/{id}'
        r = requests.get(url)
        if r.status_code != 200:
            return None
        self._bs = BeautifulSoup(r.content, 'lxml')

    def get(self, name):
        r = requests.get('https://www.cricbuzz.com/api/search/results', params={'q': name})
        if r.status_code != 200:
            return None
        resp = r.json()
        if resp is None or 'playerList' not in resp or len(resp['playerList']) == 0:
            return None
        return resp['playerList'][0]
    
    def get_info(self, id=None):
        if self._bs is None and id is None:
            return None
        elif self._bs is None and id is not None:
            self._update_soup(id)
        pattern = '.cb-lst-itm-sm:nth-child(13) , .cb-lst-itm-sm:nth-child(11) , .cb-lst-itm-sm:nth-child(9) , .cb-col-60:nth-child(7) , .cb-lst-itm-sm:nth-child(3)'
        info = [i.get_text().lower() for i in self._bs.select(pattern)]
        if len(info) < 4:
            return None
        age = info[0].split()[3][1:] if '(' in info[0] else None
        height = None
        ht_parts = info[1].split()
        # convert height from m -> ft (multiply by 3.281)
        # convert height from cm -> ft (divide by 30.48)
        if ht_parts[-1] == 'm':
            height = float(ht_parts[0]) * 3.281
        elif ht_parts[-1] == 'cm':
            height = float(ht_parts[0]) / 30.48
        elif ht_parts[-1] == 'ft':
            height = float(ht_parts[0])
        elif ht_parts[-1] == 'in':
            height = float(f'{ht_parts[0]}.{ht_parts[2]}')
        return {
            'age': age,
            'height': height,
            'role': info[2].strip(),
            'batting_style': info[3].strip(),
            'bowling_style': info[4].strip() if len(info) == 5 else ''
        }

    def get_stats(self, id=None):
        if self._bs is None and id is None:
            return None, None
        elif self._bs is None and id is not None:
            self._update_soup(id)
        num_bat_stats = 13
        pattern = '.cb-font-12 .text-right , tr~ tr+ tr .text-right , tr~ tr+ tr .cb-col-8'
        stats = [i.get_text().lower() for i in self._bs.select(pattern)]
        # strange case where for some players the table DOM structure is slightly diff
        if len(stats) < 79:
            pattern = '.cb-font-12 .text-right , tr+ tr .text-right , tr+ tr .cb-col-8'
            stats = [i.get_text().lower() for i in self._bs.select(pattern)]
        # another case where stats table is not consistent
        if len(stats) < 79:
            pattern = '.cb-plyr-thead .text-right , .cb-col-8'
            stats = [i.get_text().lower() for i in self._bs.select(pattern)]
        # discard players that have no t20 and ipl stats so far
        # NOTE: handle cases where intl / domestic players with good local t20 records
        # who might be considered for auction in reality but we are ignoring them here
        if (len(stats) == 0) or ('t20i' not in stats and 'ipl' not in stats):
            return None, None
        # extract batting stats only for t20 and ipl
        i = num_bat_stats
        bat_feat_cols = stats[:i]
        stat_idx = ['t20i', 'ipl']
        if 't20i' in stats:
            i += 1
            bat_features = [stats[i:i+num_bat_stats]]
            i += num_bat_stats
        else:
            bat_features = [['-'] * num_bat_stats]
            if len(stats) > 52:
                i += num_bat_stats + 1
        if 'ipl' in stats:
            i += 1
            bat_features.append(stats[i:i+num_bat_stats])
            i += num_bat_stats
        else:
            bat_features.append([['-'] * num_bat_stats])
            if len(stats) > 52:
                i += num_bat_stats + 1
        # extract bowl stats only for t20 and ipl
        num_bowl_stats = 12
        bowl_feat_cols = stats[i:i+num_bowl_stats]
        i += num_bowl_stats
        if 't20i' in stats:
            i += 1
            bowl_features = [stats[i:i+num_bowl_stats]]
            i += num_bowl_stats
        else:
            bowl_features = [['-'] * num_bowl_stats]
            if len(stats) > 52:
                i += num_bowl_stats + 1
        if 'ipl' in stats:
            i += 1
            bowl_features.append(stats[i:i+num_bowl_stats])
        else:
            bowl_features.append([['-'] * num_bowl_stats])
        # construct bat_df from dict
        d = {}
        for i in range(num_bat_stats):
            for j in range(len(stat_idx)):
                if bat_feat_cols[i] not in d:
                    d[bat_feat_cols[i]] = []
                d[bat_feat_cols[i]].append(bat_features[j][i])
        bat_df = pd.DataFrame(d, index=stat_idx)
        # construct bowl_df from dict
        d = {}
        for i in range(num_bowl_stats):
            for j in range(len(stat_idx)):
                if bowl_feat_cols[i] not in d:
                    d[bowl_feat_cols[i]] = []
                d[bowl_feat_cols[i]].append(bowl_features[j][i])
        bowl_df = pd.DataFrame(d, index=stat_idx)
        return bat_df, bowl_df