import sys

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path, WindowsPath
import pandas as pd

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data,
                       model, param_grid, cv=3, metric='accuracy',
                       do_probabilities=False):
        gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                scoring=metric,
                verbose=2
        )
        fitted_model = gs.fit(X_train_data, y_train_data)

        if do_probabilities:
                pred = fitted_model.predict_proba(X_test_data)
        else:
                pred = fitted_model.predict(X_test_data)

        return fitted_model, pred

if __name__ == '__main__':

        # A parameter grid for XGBoost
        params = {
                'n_estimators': [100, 300], #, 500],
                #'min_child_weight': [1, 5, 10],
                #'gamma': [0.5, 1, 2, 5],
                'subsample': [
                        0.8], #, 1.0],
                #'colsample_bytree': [0.6, 0.8], #, 1.0],
                'max_depth': [3, 4], #, 5],
                'learning_rate': [0.08, 0.1]#, 0.1]
                }

        model = XGBClassifier()  # objective='binary:logistic',silent=True, nthread=1)
        #print(Path().absolute().parents[1])
        fn = 'analog_dataset_Xy_rtus-22-26-172-228-230-11024-11027_alltime.csv'
        fp = Path(Path().absolute().parents[1], 'notebooks', fn)
        print(fp)
        if fp.is_file():
                # import dataset
                df = pd.read_csv(fp).iloc[:, 1:]
                print('dataframe dimensions: ', df.shape)
                X = df.drop(['Label'], axis=1)
                y = df['Label']
                print('X: ', X.shape, 'y: ', y.shape)

                # downsample dominating classes
                labels = ['alarm_config', 'pressure']  # ['alarm_config', 'flow', 'position', 'pressure', 'temperature']
                for l in labels:
                        y_tmp = y[y == l]
                        print('Before downsample: ', y_tmp.shape)
                        y_tmp = y_tmp.sample(frac=0.1, random_state=2021)
                        print('After downsample: ', y_tmp.shape)
                        y = pd.concat([y[y != l], y_tmp])
                        print('After downsample {}, y: {}'.format(l, y.shape))
                X = X.loc[y.index, :]
                print('X: ', X.shape)

                # normalization and split
                X_scaled = MinMaxScaler().fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=2021)
                print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
                print('Training labels: \n', y_train.value_counts())
                print('Testing labels: \n', y_test.value_counts())
                fitted_model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model,
                                                 params, cv=3)

                print(fitted_model.best_score_, '\n', fitted_model.best_params_)
        else:
                print(fp, ' not exist!')