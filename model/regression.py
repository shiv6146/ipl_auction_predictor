'''
AutoML model for team prediction and auction price prediction using auto-regressor
'''
import joblib
import pandas as pd
from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from crawlers.utils import load_data, preprocess

def train():
    df = preprocess(load_data('../data/data.csv'))
    
    # shuffle rows
    df = df.sample(frac=1)

    y_train = pd.to_numeric(df['price'], errors='coerce')
    df = df.drop(['team', 'price'], axis=1)

    # train test split
    # X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(df, y_reg, test_size=0.0, random_state=42)
    # print(f'Train Data: {(X_train_reg.shape, y_train_reg.shape)}, Test Data: {(X_test_reg.shape, y_test_reg.shape)}')
    print(f'Train Data: {(df.shape, y_train.shape)}')

    # ord cols
    ord_cols = ['country', 'role']
    # num cols
    num_cols = ['age', 'year', 'total_runs', 'total_6s', 'total_sr', 'total_wkts', 'total_bowl_econ', 'total_bowl_sr']

    # feature preprocess
    pre = ColumnTransformer([
        ('num_std_scaler', StandardScaler(), num_cols),
        ('str_ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ord_cols)
    ])

    automl = AutoSklearnRegressor(time_left_for_this_task=60, n_jobs=-1, max_models_on_disc=1, ensemble_size=50)

    # train pipeline
    pipe = Pipeline([
        ('feat_pre', pre),
        ('automl', automl)
    ],
    verbose=True)

    # train
    pipe.fit(df, y_train)

    # test
    # print(pipe.score(X_test_reg, y_test_reg))

    joblib.dump(pipe, 'auto_reg_v1.joblib')

    print(automl.leaderboard())
    print(automl.show_models())
    print(automl.sprint_statistics())

if __name__ == '__main__':
    train()