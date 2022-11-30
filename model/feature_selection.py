# example of correlation feature selection for numerical data
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from model import load_data, preprocess
from sklearn.metrics import mean_absolute_error
import numpy
 
# feature selection
def select_features(X_train, y_train, X_test, fs_method):
	# configure to select all features
	fs = SelectKBest(score_func=fs_method, k=20)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

def model_eval(model, X_test, y_test):
	# eval model
	y_hat = model.predict(X_test)
	# eval preds
	mae = mean_absolute_error(y_test, y_hat)
	print('Auto Regressor MAE: %.3f' % mae)

def model_train(X_train, y_train, X_test, y_test):
	model = LinearRegression()
	model.fit(X_train, y_train)
	# eval model
	y_hat = model.predict(X_test)
	# eval preds
	mae = mean_absolute_error(y_test, y_hat)
	print('MAE: %.3f' % mae)

df = preprocess(load_data('../data/data.csv'))
# shuffle rows
df = df.sample(frac=1)
# train test split
train_data, test_data = train_test_split(df, test_size=0.1)
# print(f'Train Data: {train_data.shape}, Test Data: {test_data.shape}')
# Drop y_target column
y_train = train_data.price
X_train = train_data.drop(['price'], axis=1)
# predict on test data
y_test = test_data.price
X_test = test_data.drop(['price'], axis=1)
# model train with all features
model_train(X_train, y_train, X_test, y_test)
# load autosklearn.regressor model
auto_reg = joblib.load('auto_regressor.joblib')
model_eval(auto_reg, X_test, y_test)
# feature selection f_regression correlation features top 20
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_regression)
# model train with top 20 correlation features
model_train(X_train_fs, y_train, X_test_fs, y_test)
model_eval(auto_reg, X_test_fs, y_test)
# feature selection mutual_info features top 20
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_regression)
# model train with top 20 mutual_info features
model_train(X_train_fs, y_train, X_test_fs, y_test)
model_eval(auto_reg, X_test_fs, y_test)

y_train = df.price
X_train = df.drop(['price'], axis=1)
# define the evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LinearRegression()
fs = SelectKBest(score_func=mutual_info_regression)
pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])

# define the grid
grid = dict()
grid['sel__k'] = [i for i in range(X_train.shape[1]-20, X_train.shape[1]+1)]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X_train, y_train)
# summarize best
print('Best MAE: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))

# define number of features to evaluate
num_features = [i for i in range(X_train.shape[1]-15, X_train.shape[1]+1)]
# enumerate each number of features
results = list()
for k in num_features:
	# create pipeline
	model = LinearRegression()
	fs = SelectKBest(score_func=mutual_info_regression, k=k)
	pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
	# evaluate the model
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
	results.append(scores)
	# summarize the results
	print(f'{k}, {numpy.mean(scores)}, {numpy.std(scores)}')
# plot model performance for comparison
pyplot.boxplot(results, labels=num_features, showmeans=True)
pyplot.show()
# what are scores for the features
# for i in range(len(fs.scores_)):
# 	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()