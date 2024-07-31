import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier

## Funktion zur Erstellung der simulierten Daten
def create_surv_data(shape_weibull=1.5, scale_weibull_base=50, rate_censoring=0.02, n=1000, 
                     b_bloodp=-0.405, b_diab=0.4, b_age=0.05, b_bmi=0.01, b_kreat=0.2, seed=42) -> pd.DataFrame:

    # Parameter für Weibull-Verteilung und Censoring
    shape_weibull = shape_weibull
    scale_weibull_base = scale_weibull_base
    rate_censoring = rate_censoring
    n = n

    # Generierung der Kovariaten
    np.random.seed(seed)
    bmi = np.random.normal(25, 5, n)  
    blood_pressure = np.random.binomial(1, 0.3, n)  
    kreatinkinase = np.random.lognormal(mean=5, sigma=1, size=n)
    kreatinkinase = np.clip(kreatinkinase, 30, 8000)
    diabetes = np.random.binomial(1, 0.2, n)  
    age = np.random.normal(50, 10, n)  #
    #plt.boxplot(kreatinkinase)
    #plt.show()

    # Parameter für Weibull-Verteilung
    lambda_weibull = scale_weibull_base * np.exp(
        b_bloodp * blood_pressure +         # Linearer Einfluss von hohem Blutdruck
        b_diab * diabetes +                 # Linearer Einfluss von Diabetes
        b_age * age +                       # Linearer Einfluss des Alters
        b_bmi * (bmi - 25) ** 2  +          # Quadratischer Einfluss des BMI
        b_kreat * np.log(kreatinkinase )    # Exponentieller Einfluss der Kreatinkinase
    )

    # Generierung der Ereigniszeiten basierend auf der Weibull-Verteilung
    event_times = np.random.weibull(shape_weibull, n) * lambda_weibull
    censoring_times = np.random.exponential(1 / rate_censoring, n)
    observed_times = np.minimum(event_times, censoring_times)
    event_occurred = event_times <= censoring_times

    # Erstellung des Datensatzes ohne die nicht-linearen Transformationen
    data = pd.DataFrame({
        'bmi': bmi,
        'blood_pressure': blood_pressure.astype(int), 
        'kreatinkinase': kreatinkinase,
        'diabetes': diabetes.astype(int),
        'age': age,
        't': observed_times,
        'event': event_occurred.astype(int)
    })
    
    print('Data shape:', data.shape)
    print(f'{(data["event"] ==1).sum()/n  * 100} % of the data has an event')
    
    return pd.DataFrame(data)



def create_new_dataset_with_ipcw_weights(data: pd.DataFrame, t: np.float64, kmf: KaplanMeierFitter) -> pd.DataFrame:
    new_data = data.copy()

    new_data.loc[(data['time'] <= t) & (data['event'] == 1), 'survived'] = int(0)
    new_data.loc[(data['time'] >= t) & (data['event'] == 0), 'survived'] = int(1)
    new_data.loc[(data['time'] >= t) & (data['event'] == 1), 'survived'] = int(1)
    new_data.loc[(data['time'] <= t) & (data['event'] == 0), 'survived'] = int(999)

    new_data['survived'] = new_data['survived'].astype(int)

    ipcw_weights =  1/kmf.survival_function_at_times(new_data['time'])
    ipcw_weight_tau = 1/kmf.survival_function_at_times(t)
    new_data['weights_ipcw'] = np.where(new_data['survived']==1, ipcw_weight_tau, np.where(new_data['survived']==0, ipcw_weights, 0))
    new_data['weights_ipcw'] = new_data['weights_ipcw'] / new_data['weights_ipcw'].sum()

    return pd.DataFrame(new_data)



def train_test_split_into_df(df_train: pd.DataFrame, df_test: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    y_train_df = pd.DataFrame({'event': [event for event, time in y_train], 'time': [time for event, time in y_train]})
    y_test_df = pd.DataFrame({'event': [event for event, time in y_test], 'time': [time for event, time in y_test]})

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train['event'] = y_train_df['event']
    df_train['time'] = y_train_df['time']
    df_test['event'] = y_test_df['event']
    df_test['time'] = y_test_df['time']

    return df_train, df_test

class BootstrapRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha,
                         max_samples=max_samples)
        self.tree_predictions_ = []

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        super().fit(X, y, sample_weight)
        self._store_tree_predictions(X)
        return self

    def _store_tree_predictions(self, X):
        for estimator in self.estimators_:
            self.tree_predictions_.append(estimator.predict_proba(X)[:, 1])

    def get_bootstrap_samples_and_predictions(self):
        return [(indices, predictions) for indices, predictions in zip(self.estimators_samples_, self.tree_predictions_)]

