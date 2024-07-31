import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter, WeibullAFTFitter
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def create_surv_data(
    shape_weibull=1.5,
    scale_weibull_base=50,
    rate_censoring=0.02,
    n=1000,
    b_bloodp=-0.405,
    b_diab=0.4,
    b_age=0.05,
    b_bmi=0.01,
    b_kreat=0.2,
    seed=42,
) -> pd.DataFrame:
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

    # Parameter für Weibull-Verteilung
    lambda_weibull = scale_weibull_base * np.exp(
        b_bloodp * blood_pressure
        + b_diab * diabetes  # Linearer Einfluss von hohem Blutdruck
        + b_age * age  # Linearer Einfluss von Diabetes
        + b_bmi * (bmi - 25) ** 2  # Linearer Einfluss des Alters
        + b_kreat  # Quadratischer Einfluss des BMI
        * np.log(kreatinkinase)  # Exponentieller Einfluss der Kreatinkinase
    )

    # Generierung der Ereigniszeiten basierend auf der Weibull-Verteilung
    event_times = np.random.weibull(shape_weibull, n) * lambda_weibull
    censoring_times = np.random.exponential(1 / rate_censoring, n)
    observed_times = np.minimum(event_times, censoring_times)
    event_occurred = event_times <= censoring_times

    # Erstellung des Datensatzes ohne die nicht-linearen Transformationen
    data = pd.DataFrame(
        {
            "bmi": bmi,
            "blood_pressure": blood_pressure.astype(int),
            "kreatinkinase": kreatinkinase,
            "diabetes": diabetes.astype(int),
            "age": age,
            "t": observed_times,
            "event": event_occurred.astype(int),
        }
    )

    print("Data shape:", data.shape)
    print(f'{(data["event"] ==1).sum()/n  * 100} % of the data has an event')

    return pd.DataFrame(data)


def create_new_dataset_with_ipcw_weights(
    data: pd.DataFrame, t: np.float64, kmf: KaplanMeierFitter
) -> pd.DataFrame:
    new_data = data.copy()

    new_data.loc[(data["time"] <= t) & (data["event"] == 1), "survived"] = int(0)
    new_data.loc[(data["time"] >= t) & (data["event"] == 0), "survived"] = int(1)
    new_data.loc[(data["time"] >= t) & (data["event"] == 1), "survived"] = int(1)
    new_data.loc[(data["time"] <= t) & (data["event"] == 0), "survived"] = int(999)

    new_data["survived"] = new_data["survived"].astype(int)

    ipcw_weights = 1 / kmf.survival_function_at_times(new_data["time"])
    ipcw_weight_tau = 1 / kmf.survival_function_at_times(t)
    new_data["weights_ipcw"] = np.where(
        new_data["survived"] == 1,
        ipcw_weight_tau,
        np.where(new_data["survived"] == 0, ipcw_weights, 0),
    )
    new_data["weights_ipcw"] = new_data["weights_ipcw"] / new_data["weights_ipcw"].sum()

    return pd.DataFrame(new_data)


def train_test_split_into_df(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    # Reset index of train and test DataFrames
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    # Convert y_train and y_test to DataFrames
    y_train_df = pd.DataFrame(y_train, columns=["event", "time"])
    y_test_df = pd.DataFrame(y_test, columns=["event", "time"])

    # Assign event and time columns to train and test DataFrames
    df_train[["event", "time"]] = y_train_df[["event", "time"]]
    df_test[["event", "time"]] = y_test_df[["event", "time"]]

    return df_train, df_test


def create_train_test_data(params: dict) -> pd.DataFrame:

    ### Parameter für Weibull-Verteilung und Censoring ###
    shape_weibull = params.get('shape_weibull')
    scale_weibull_base = params.get('scale_weibull_base')
    rate_censoring = params.get('rate_censoring')
    n = params.get('n')
    b_bloodp = params.get('b_bloodp')
    b_diab = params.get('b_diab')
    b_age = params.get('b_age')
    b_bmi = params.get('b_bmi')
    b_kreat = params.get('b_kreat')
    seed = params.get('seed')
    tau = params.get('tau')

    ### Generierung der Kovariaten ###
    np.random.seed(seed)
    bmi = np.random.normal(25, 5, n)
    blood_pressure = np.random.binomial(1, 0.3, n)
    kreatinkinase = np.random.lognormal(mean=5, sigma=1, size=n)
    kreatinkinase = np.clip(kreatinkinase, 30, 8000)
    diabetes = np.random.binomial(1, 0.2, n)
    age = np.random.normal(50, 10, n)  #

    ### Weibull-Verteilung ###
    lambda_weibull = scale_weibull_base * np.exp(
        b_bloodp * blood_pressure
        + b_diab * diabetes  # Linearer Einfluss von hohem Blutdruck
        + b_age * age  # Linearer Einfluss von Diabetes
        + b_bmi * (bmi - 25) ** 2  # Linearer Einfluss des Alters
        + b_kreat  # Quadratischer Einfluss des BMI
        * np.log(kreatinkinase)  # Exponentieller Einfluss der Kreatinkinase
    )

    ### Generierung der Ereigniszeiten/Zensierzeiten basierend auf der Weibull-/ZensierVerteilung
    event_times = np.random.weibull(shape_weibull, n) * lambda_weibull
    censoring_times = np.random.exponential(1 / rate_censoring, n)
    observed_times = np.minimum(event_times, censoring_times)
    event_occurred = event_times <= censoring_times

    ### Erstellung des Datensatzes ohne die Transformationen ###
    data = pd.DataFrame(
        {
            "bmi": bmi,
            "blood_pressure": blood_pressure.astype(int),
            "kreatinkinase": kreatinkinase,
            "diabetes": diabetes.astype(int),
            "age": age,
            "t": observed_times,
            "event": event_occurred.astype(int),
        }
    )
    #print("Data shape:", data.shape)
    #print(f'{(data["event"] ==1).sum()/n  * 100} % of the data has an event')


    ### Startified Split ###
    X = data[['bmi', 'blood_pressure', 'kreatinkinase', 'diabetes', 'age']]
    y = Surv.from_arrays(event=data['event'], time=data['t'])
    df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    df_train, df_test = train_test_split_into_df(df_train=df_train, df_test=df_test, y_train=y_train, y_test=y_test)


    ### cut data at tau // ipcw weights ###
    kmf = KaplanMeierFitter()
    kmf.fit(df_train['time'], event_observed=1-df_train['event'])
    df_train = create_new_dataset_with_ipcw_weights(data=df_train,t=tau, kmf=kmf)
    df_test = create_new_dataset_with_ipcw_weights(data=df_test,t=tau, kmf=kmf)

    portions_at_cutpoint = df_train['survived'].value_counts(normalize=True)
    portion_censored_after_cut_train = portions_at_cutpoint[999]
    n_events_after_cut_train = portions_at_cutpoint[1] * df_train.shape[0] 

    return df_train, df_test, n_events_after_cut_train, portion_censored_after_cut_train


def ipc_weighted_mse(y_true, y_pred, sample_weight):
    return np.average((y_true - y_pred) ** 2, weights=sample_weight)


def get_Nbi(lists):
    arr = np.array(lists)
    max_value = arr.max()
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=max_value + 1), axis=1, arr=arr
    )
    return counts


def inf_JK_bagged_variance(
    N_bi: np.ndarray, T_N_b: np.ndarray, weights: np.ndarray = None
):
    B, n = N_bi.shape
    T_N_b_mean = np.mean(T_N_b, axis=0)
    m = np.count_nonzero(weights)

    cov_i = ((N_bi - n * weights[0]).T @ (T_N_b - T_N_b_mean)) / B
    cov_i_hoch2 = cov_i**2
    biased_var_estimate = np.sum(cov_i_hoch2, axis=0)

    bias_correction = n / B * (m - 1) / m * np.var(T_N_b, axis=0)

    return biased_var_estimate, bias_correction


def simulation(n:int, seed:int, sims: int, tau:float, data_generation_weibull_parameters:dict  ):

    df_train, df_test, n_events_after_cut_train, portion_censored_after_cut_train = create_train_test_data(data_generation_weibull_parameters)

