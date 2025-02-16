import sys
sys.path.append('../')
import joblib
from autosklearn.classification import AutoSklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from crawlers.utils import load_data, preprocess

def train():
    df = preprocess(load_data('../data/data.csv'))
    
    # shuffle rows
    df = df.sample(frac=1)

    y_train = df['team']
    df = df.drop(['team', 'price'], axis=1)

    # train test split
    # X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(df, y_clf, test_size=0.1, random_state=42)
    # print(f'Train Data: {(X_train_clf.shape, y_train_clf.shape)}, Test Data: {(X_test_clf.shape, y_test_clf.shape)}')
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

    automl = AutoSklearnClassifier(time_left_for_this_task=600, per_run_time_limit=60, n_jobs=-1, max_models_on_disc=50, ensemble_size=50)
    # rf = RandomForestClassifier(verbose=2, n_jobs=-1)

    # train pipeline
    pipe = Pipeline([
        ('feat_pre', pre),
        ('auto_clf', automl)
    ],
    verbose=True)

    # train
    pipe.fit(df, y_train)

    # test
    # print(pipe.score(X_test_clf, y_test_clf))

    joblib.dump(pipe, 'auto_clf_v1.joblib')

    print(y_train.head())
    print(pipe.predict(df.head()))

    print(automl.leaderboard())
    print(automl.show_models())
    print(automl.sprint_statistics())

if __name__ == '__main__':
    train()