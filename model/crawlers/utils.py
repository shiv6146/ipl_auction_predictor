import os
import pandas as pd
from .cricbuzz import Player

def get_player_features(name, year):
    player = None
    try:
        player = Player(name)
    except Exception as e:
        return None
    if player is None or player.info is None or player.bat_stats is None or player.bowl_stats is None:
        return None
    # feature vector column names
    feature_names = [
        "name", "country", "age", "height", "role", "bat_style", "bowl_style", "t20_no", "t20_runs", "t20_avg", "t20_sr", "t20_50", "t20_4s", "t20_6s",
        "ipl_no", "ipl_runs", "ipl_avg", "ipl_sr", "ipl_50", "ipl_4s", "ipl_6s", "t20_wkts", "t20_bowl_econ", "t20_bowl_avg", "t20_bowl_sr",
        "ipl_wkts", "ipl_bowl_econ", "ipl_bowl_avg", "ipl_bowl_sr", "year"
    ]
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
    features.append(year)
    return preprocess(pd.DataFrame([features], columns=feature_names))

def load_data(data_file: str) -> pd.DataFrame:
    if not os.path.isfile(data_file):
        return None
    return pd.read_csv(data_file)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = None
    # Drop unwanted columns
    processed_df = df.drop(['name', 'height', 'bat_style', 'bowl_style', 't20_no', 't20_avg', 't20_50', 
    't20_4s', 'ipl_no', 'ipl_avg', 'ipl_50', 'ipl_4s', 't20_bowl_avg', 'ipl_bowl_avg'], axis=1)
    # convert all stats column to numeric
    processed_df = processed_df.replace('-', 0.0)
    for col in processed_df.columns[3:15]:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
    # merge columns together
    processed_df['total_runs'] = processed_df['t20_runs'] + processed_df['ipl_runs']
    processed_df['total_6s'] = processed_df['t20_6s'] + processed_df['ipl_6s']
    processed_df['total_sr'] = processed_df['t20_sr'] / 2 + processed_df['ipl_sr'] / 2
    processed_df['total_wkts'] = processed_df['t20_wkts'] + processed_df['ipl_wkts']
    processed_df['total_bowl_econ'] = processed_df['t20_bowl_econ'] / 2 + processed_df['ipl_bowl_econ'] / 2
    processed_df['total_bowl_sr'] = processed_df['t20_bowl_sr'] / 2 + processed_df['ipl_bowl_sr'] / 2

    # drop all merged cols
    processed_df = processed_df.drop(['t20_runs', 'ipl_runs', 't20_6s', 'ipl_6s', 'ipl_sr', 
    't20_sr', 'ipl_wkts', 't20_wkts', 'ipl_bowl_econ', 't20_bowl_econ', 't20_bowl_sr', 'ipl_bowl_sr'], axis=1)
    
    return processed_df