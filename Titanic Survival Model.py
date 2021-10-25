import pandas as pd
import pickle

from sklearn.metrics import classification_report, roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from helpers.eda import *
from helpers.data_prep import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def load():
    data = pd.read_csv("titanic.csv")
    return data


df = load()
check_df(df)

df["Age"].hist(bins=50)
plt.show()

cat, num, cat_b_car = grab_col_names(df)

for col in cat:
    target_summary_with_cat(df, "Survived", col)

for col in num:
    target_summary_with_num (df, "Survived", col)

for col in cat:
    cat_summary(df, col, plot=True)


def titanic_data_prep(data):
    dataframe = data.copy ()

    # Feature Engineering
    dataframe.columns = [col.upper() for col in dataframe.columns]
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # MISSING VALUES
    missing_cols = [col for col in dataframe.columns if (dataframe[col].isnull().any()) & (col != "Cabin")]
    for i in missing_cols:
        if i == "AGE":
            dataframe[i].fillna(dataframe.groupby("PCLASS")[i].transform("median"), inplace=True)
        elif dataframe[i].dtype == "O":
            dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)
        else:
            dataframe[i].fillna(dataframe[i].median(), inplace=True)

    # Outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    deleted_cols = ["CABIN", "SIBSP", "PARCH", "TICKET", "NAME"]
    dataframe = dataframe.drop(deleted_cols, axis=1)

    dataframe["NEW_AGE_CAT"] = pd.cut(dataframe["AGE"], bins=[0, 20, 35, 55, dataframe["AGE"].max() + 1],
                                       labels=[1, 2, 3, 4]).astype(int)

    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 20), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
            (dataframe['AGE'] > 20) & (dataframe['AGE']) < 55), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 20), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
            (dataframe['AGE'] > 20) & (dataframe['AGE']) < 55), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 55), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe["NEW_PCLASS_AGE"] = dataframe["PCLASS"] / (dataframe["AGE"].astype(int) + 1)
    dataframe["NEW_FARE_AGE"] = dataframe["FARE"] / (dataframe["AGE"].astype(int) + 1)

    cat_cols, num_cols, cat_but_car = grab_col_names (dataframe)

    # Label Encoding
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Rare Encoding
    dataframe = rare_encoder(dataframe, 0.01, cat_cols)

    # One-Hot Encoding
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique () > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # Standart Scaler
    scaler = StandardScaler ()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe


df.head()
df.describe()

df_prep = titanic_data_prep(df)

df_prep.head()
df_prep.describe()
check_df(df_prep)
df_prep.isnull().sum().sum()

# Model
y = df_prep["SURVIVED"]
X = df_prep.drop (["PASSENGERID", "SURVIVED"], axis=1)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_pred)

plot_confusion_matrix(rf_model, X_test, y_test)
plt.show();

# MODEL TUNING
rf_params = {"n_estimators": [1000, 1200],
             "max_depth": [10, 12],
             "min_samples_split": [2, 3, 5]}

rf = RandomForestClassifier(random_state=46)
rf_cv = GridSearchCV(rf, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)

rf_cv.best_params_

# FINAL MODEL
rf_tuned = rf.set_params (**rf_cv.best_params_, random_state=46).fit(X_train, y_train)
y_final_pred = rf_tuned.predict(X_test)

print (classification_report(y_test, y_final_pred))
roc_auc_score(y_test, y_final_pred)

plot_confusion_matrix(rf_tuned, X_test, y_test)
plt.show ();

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X_train)