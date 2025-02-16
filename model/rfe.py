import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from crawlers.utils import load_data, preprocess

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()
 
# get the dataset
def get_dataset():
	df = preprocess(load_data('../data/data.csv'))
	df.sample(frac=1)
	df.dropna(inplace=True)
	categorical_cols = ['country', 'role']
	num_cols = ['age', 'year', 'total_runs', 'total_6s', 'total_sr', 'total_wkts', 'total_bowl_econ', 'total_bowl_sr']
	df[num_cols] = scaler.fit_transform(df[num_cols])
	df[categorical_cols] = enc.fit_transform(df[categorical_cols])
	X_train = df.drop(['team', 'price'], axis=1)
	y_train = df.team
	print(df['team'].value_counts(normalize=True) * 100)
	return X_train, y_train
 
# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(5, 24):
		rfe = RFE(estimator=DecisionTreeClassifier(class_weight='balanced'), n_features_to_select=i)
		model = DecisionTreeClassifier(class_weight='balanced')
		models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 
# define dataset
X, y = get_dataset()
print(X.shape, y.shape)
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()