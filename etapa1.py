import pandas as pd
import seaborn as sns
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, precision_recall_fscore_support


# Citim datele din fisierul arff
# data_train, meta_train = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
#                                        "1/RacketSports/RacketSports_TRAIN.arff")
# data_test, meta_test = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
#                                      "1/RacketSports/RacketSports_TEST.arff")
data_train1, meta_train1 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                         "1/RacketSports/RacketSportsDimension1_TRAIN.arff")
data_test1, meta_test1 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                       "1/RacketSports/RacketSportsDimension1_TEST.arff")
data_train2, meta_train2 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                         "1/RacketSports/RacketSportsDimension2_TRAIN.arff")
data_test2, meta_test2 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                       "1/RacketSports/RacketSportsDimension2_TEST.arff")
data_train3, meta_train3 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                         "1/RacketSports/RacketSportsDimension3_TRAIN.arff")
data_test3, meta_test3 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                       "1/RacketSports/RacketSportsDimension3_TEST.arff")
data_train4, meta_train4 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                         "1/RacketSports/RacketSportsDimension4_TRAIN.arff")
data_test4, meta_test4 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                       "1/RacketSports/RacketSportsDimension4_TEST.arff")
data_train5, meta_train5 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                         "1/RacketSports/RacketSportsDimension5_TRAIN.arff")
data_test5, meta_test5 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                       "1/RacketSports/RacketSportsDimension5_TEST.arff")
data_train6, meta_train6 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                         "1/RacketSports/RacketSportsDimension6_TRAIN.arff")
data_test6, meta_test6 = arff.loadarff("C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa "
                                       "1/RacketSports/RacketSportsDimension6_TEST.arff")

# Transformam datele intr-un dataframe pandas si convertim coloana 'activity' la string
# df_train = pd.DataFrame(data_train)
# df_test = pd.DataFrame(data_test)
df_train1 = pd.DataFrame(data_train1)
df_test1 = pd.DataFrame(data_test1)
df_train2 = pd.DataFrame(data_train2)
df_test2 = pd.DataFrame(data_test2)
df_train3 = pd.DataFrame(data_train3)
df_test3 = pd.DataFrame(data_test3)
df_train4 = pd.DataFrame(data_train4)
df_test4 = pd.DataFrame(data_test4)
df_train5 = pd.DataFrame(data_train5)
df_test5 = pd.DataFrame(data_test5)
df_train6 = pd.DataFrame(data_train6)
df_test6 = pd.DataFrame(data_test6)

# Lista cu titlurile coloanelor
col_names = ['c'+str(i) for i in range(0, 30)]
col_names.append('activity')

# Renumește coloanele din toate dataframe-urile
df_train1 = df_train1.rename(columns=dict(zip(df_train1.columns, col_names)))
df_train2 = df_train2.rename(columns=dict(zip(df_train2.columns, col_names)))
df_train3 = df_train3.rename(columns=dict(zip(df_train3.columns, col_names)))
df_train4 = df_train4.rename(columns=dict(zip(df_train4.columns, col_names)))
df_train5 = df_train5.rename(columns=dict(zip(df_train5.columns, col_names)))
df_train6 = df_train6.rename(columns=dict(zip(df_train6.columns, col_names)))
df_test1 = df_test1.rename(columns=dict(zip(df_test1.columns, col_names)))
df_test2 = df_test2.rename(columns=dict(zip(df_test2.columns, col_names)))
df_test3 = df_test3.rename(columns=dict(zip(df_test3.columns, col_names)))
df_test4 = df_test4.rename(columns=dict(zip(df_test4.columns, col_names)))
df_test5 = df_test5.rename(columns=dict(zip(df_test5.columns, col_names)))
df_test6 = df_test6.rename(columns=dict(zip(df_test6.columns, col_names)))

# Concatenez train
dfs_train = [df_train1, df_train2, df_train3, df_train4, df_train5, df_train6]
df_train_all = pd.concat(dfs_train, axis=0, ignore_index=True)

# Concatenez test
dfs_test = [df_test1, df_test2, df_test3, df_test4, df_test5, df_test6]
df_test_all = pd.concat(dfs_test, axis=0, ignore_index=True)

# Adaugam o noua coloana "dataset" in dataframe-urile df_train si df_test si le setam valorile corespunzatoare.
# Apoi concatenam cele doua dataframe-uri in df_total.
df_train_all['dataset'] = 'train'
df_test_all['dataset'] = 'test'

col_names = df_train_all.columns
df_test_all = df_test_all.rename(columns=dict(zip(df_test_all.columns, col_names)))
df_total = pd.concat([df_train_all, df_test_all], axis=0, ignore_index=True)

# Countplot pentru seturile de antrenare si testare combinate, afisand frecventa fiecarei etichete/clase.
plt.figure(figsize=(8, 6))
sns.countplot(x='activity', hue='dataset', data=df_total)
plt.xlabel('Etichete (clase)')
plt.ylabel('Numar de aparitii')
plt.title('Frecventa de aparitie a fiecarei etichete (clase) in seturile de date de antrenare si testare')
# plt.show()




# Citim setul de date MITBIH
mitbih_train = pd.read_csv('C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa 1/archive/mitbih_train.csv', header=None)
mitbih_test = pd.read_csv('C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa 1/archive/mitbih_test.csv', header=None)
#
# # Numărăm numărul de exemple pentru fiecare clasă în ambele seturi de date
# mitbih_train_count = mitbih_train[187].value_counts().sort_index()
# mitbih_test_count = mitbih_test[187].value_counts().sort_index()
#
# # Creeăm o listă cu numărul de exemple pentru fiecare etichetă, pentru ambele seturi de date
# mitbih_counts = [mitbih_train_count.values, mitbih_test_count.values]
#
# # Definim numărul de etichete
# mitbih_num_labels = len(mitbih_train_count)
#
# # Definim lățimea barelor
# bar_width = 0.35
#
# # Definim poziția barelor pe axa x
# mitbih_train_pos = np.arange(mitbih_num_labels)
# mitbih_test_pos = mitbih_train_pos + bar_width
#
# # Generăm barplot-ul
# plt.bar(mitbih_train_pos, mitbih_counts[0], bar_width, label='Train')
# plt.bar(mitbih_test_pos, mitbih_counts[1], bar_width, label='Test')
# plt.title('Frecventa etichetelor in seturile de date MITBIH')
# plt.xlabel('Eticheta')
# plt.ylabel('Numar de exemple')
# plt.xticks(mitbih_train_pos + bar_width/2, mitbih_train_count.index)
# plt.legend()
# plt.show()





# Citim setul de date PTBDB
ptbdb_normal = pd.read_csv('C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa 1/archive/ptbdb_normal.csv', header=None)
ptbdb_abnormal = pd.read_csv('C:/Users/adean/OneDrive/Desktop/Facultate/ML/Etapa 1/archive/ptbdb_abnormal.csv', header=None)

# Concatenăm cele două seturi de date
ptbdb = pd.concat([ptbdb_normal, ptbdb_abnormal])

# Împărțim datele în train și test
ptbdb_train, ptbdb_test = train_test_split(ptbdb, test_size=0.2, random_state=42)
#
# # Numărăm numărul de exemple pentru fiecare clasă în ambele seturi de date
# ptbdb_train_count = ptbdb_train[187].value_counts().sort_index()
# ptbdb_test_count = ptbdb_test[187].value_counts().sort_index()
#
# # Creeăm o listă cu numărul de exemple pentru fiecare etichetă, pentru ambele seturi de date
# ptbdb_counts = [ptbdb_train_count.values, ptbdb_test_count.values]
#
# # Definim numărul de etichete
# ptbdb_num_labels = len(ptbdb_train_count)
#
# # Definim lățimea barelor
# bar_width = 0.35
#
# # Definim poziția barelor pe axa x
# ptbdb_train_pos = np.arange(ptbdb_num_labels)
# ptbdb_test_pos = ptbdb_train_pos + bar_width
#
# # Generăm barplot-ul
# plt.bar(ptbdb_train_pos, ptbdb_counts[0], bar_width, label='Train')
# plt.bar(ptbdb_test_pos, ptbdb_counts[1], bar_width, label='Test')
# plt.title('Frecventa etichetelor in seturile de date PTBDB')
# plt.xlabel('Eticheta')
# plt.ylabel('Numar de exemple')
# plt.xticks(ptbdb_train_pos + bar_width/2, ptbdb_train_count.index)
# plt.legend()
# plt.show()




# # valorile de accelerometru pe dimensiunile x, y și z
# df_train1_grouped = df_train1.groupby('activity')
# df_train1_specific = df_train1_grouped.first().reset_index()
# df_train2_grouped = df_train2.groupby('activity')
# df_train2_specific = df_train2_grouped.first().reset_index()
# df_train3_grouped = df_train3.groupby('activity')
# df_train3_specific = df_train3_grouped.first().reset_index()
#
# df_train_acc = pd.concat([df_train1_specific, df_train2_specific, df_train3_specific], axis=0)
#
# # Extrag numele activitatilor
# activities = df_train_acc['activity'].unique()
#
# for activity in activities:
#     fig, ax = plt.subplots()
#     ax.set_title("Time series for racket sports ({})".format(activity.decode('utf-8')))
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Accelerometer values')
#
#     for i in range(3):
#         label = 'c{}'.format(i)
#         df = df_train_acc[df_train_acc['activity'] == activity].iloc[:, 1:]
#         series = df.iloc[i]
#         ax.plot(series, label=label)
#
#     ax.legend(['x', 'y', 'z'])
#     plt.show()
#
#
#
#
#
#
# # valorile de giroscop pe dimensiunile x, y și z
# df_train4_grouped = df_train4.groupby('activity')
# df_train4_specific = df_train4_grouped.first().reset_index()
# df_train5_grouped = df_train5.groupby('activity')
# df_train5_specific = df_train5_grouped.first().reset_index()
# df_train6_grouped = df_train6.groupby('activity')
# df_train6_specific = df_train6_grouped.first().reset_index()
#
# df_train_gir = pd.concat([df_train4_specific, df_train5_specific, df_train6_specific], axis=0)
#
# # Extrag numele activitatilor
# activities = df_train_gir['activity'].unique()
#
# for activity in activities:
#     fig, ax = plt.subplots()
#     ax.set_title("Time series for racket sports ({})".format(activity.decode('utf-8')))
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Giroscope values')
#
#     for i in range(3):
#         label = 'c{}'.format(i)
#         df = df_train_gir[df_train_gir['activity'] == activity].iloc[:, 1:]
#         series = df.iloc[i]
#         ax.plot(series, label=label)
#
#     ax.legend(['x', 'y', 'z'])
#     plt.show()




#
# # câte un exemplu de serie pentru fiecare categorie de aritmie pentru MITBIH
# mitbih_train_grouped = mitbih_train.groupby(by=[187])
# mitbih_train_specific = mitbih_train_grouped.first().reset_index()
#
# # pentru fiecare linie din dataframe
# for i in range(len(mitbih_train_specific)):
#     # folosim coloana 187 pentru a da numele seriei
#     label = 'aritmia ' + str(int(mitbih_train_specific.iloc[i, 0]))
#     # folosim restul valorilor de pe linie pentru a crea seria
#     data = mitbih_train_specific.iloc[i, 1:]
#     # plottam seria
#     plt.plot(data, label=label)
#
#     # adaugam titluri și legenda
#     plt.title('Exemplu de serie pentru MITBIH')
#     plt.xlabel('Timp')
#     plt.ylabel('Amplitudine')
#     plt.legend()
#
#     # afisam graficul
#     plt.show()
#
#
# # câte un exemplu de serie pentru fiecare categorie de aritmie pentru PTBDB
# ptbdb_train_grouped = ptbdb_train.groupby(by=[187])
# ptbdb_train_specific = ptbdb_train_grouped.first().reset_index()
#
# # pentru fiecare linie din dataframe
# for i in range(len(ptbdb_train_specific)):
#     # folosim coloana 187 pentru a da numele seriei
#     label = 'aritmia ' + str(int(ptbdb_train_specific.iloc[i, 0]))
#     # folosim restul valorilor de pe linie pentru a crea seria
#     data = ptbdb_train_specific.iloc[i, 1:]
#     # plottam seria
#     plt.plot(data, label=label)
#
#     # adaugam titluri și legenda
#     plt.title('Exemplu de serie pentru PTBDB')
#     plt.xlabel('Timp')
#     plt.ylabel('Amplitudine')
#     plt.legend()
#
#     # afisam graficul
#     plt.show()





# # Media si deviatia standard pentru MITBIH
# # Concatenăm datele de antrenament și testare pentru MITBIH
# mitbih = pd.concat([mitbih_train, mitbih_test])
#
# # Definim numele coloanelor pentru setul de date MITBIH
# columns = list(range(187))
# columns.append('label')
# mitbih.columns = columns
#
# mitbih_arithmias = mitbih_train[187].unique()
#
# # Iterăm prin fiecare aritmie
# for i in range(len(mitbih_arithmias)):
#     # Selectăm doar exemplele cu aritmia i
#     data = mitbih[mitbih['label'] == i]
#
#     # Calculăm media și deviația standard per unitate de timp
#     mean = data.mean(axis=0)
#     std = data.std(axis=0)
#
#     # Afișăm graficul pentru media și deviația standard
#     fig, ax = plt.subplots()
#     ax.plot(mean[:-1], label='mean')
#     ax.plot(std[:-1], label='std')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.set_title(f'MITBIH Arrhythmia {i}')
#     ax.legend()
#     plt.show()





# # Media si deviatia standard pentru PTBDB
# # Concatenăm datele de antrenament și testare pentru PTBDB
# ptbdb = pd.concat([ptbdb_train, ptbdb_test])
#
# # Definim numele coloanelor pentru setul de date PTBDB
# columns = list(range(187))
# columns.append('label')
# ptbdb.columns = columns
#
# ptbdb_arithmias = ptbdb_train[187].unique()
#
# # Iterăm prin fiecare aritmie
# for i in range(len(ptbdb_arithmias)):
#     # Selectăm doar exemplele cu aritmia i
#     data = ptbdb[ptbdb['label'] == i]
#
#     # Calculăm media și deviația standard per unitate de timp
#     mean = data.mean(axis=0)
#     std = data.std(axis=0)
#
#     # Afișăm graficul pentru media și deviația standard
#     fig, ax = plt.subplots()
#     ax.plot(mean[:-1], label='mean')
#     ax.plot(std[:-1], label='std')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.set_title(f'PTBDB Arrhythmia {i}')
#     ax.legend()
#     plt.show()





# # distribuția valorilor per fiecare axă de accelerometru și giroscop în parte / per acțiune (RACKETSPORTS)
#
# dfs1 = [df_train1, df_test1]
# df1 = pd.concat(dfs1, axis=0, ignore_index=True)
# dfs2 = [df_train2, df_test2]
# df2 = pd.concat(dfs2, axis=0, ignore_index=True)
# dfs3 = [df_train3, df_test3]
# df3 = pd.concat(dfs3, axis=0, ignore_index=True)
# dfs4 = [df_train4, df_test4]
# df4 = pd.concat(dfs4, axis=0, ignore_index=True)
# dfs5 = [df_train5, df_test5]
# df5 = pd.concat(dfs5, axis=0, ignore_index=True)
# dfs6 = [df_train6, df_test6]
# df6 = pd.concat(dfs6, axis=0, ignore_index=True)
#
# sns.FacetGrid(df1, hue = 'activity', height = 4).map(sns.distplot, 'c13', kde=True).add_legend()
# sns.FacetGrid(df2, hue = 'activity', height = 4).map(sns.distplot, 'c13', kde=True).add_legend()
# sns.FacetGrid(df3, hue = 'activity', height = 4).map(sns.distplot, 'c13', kde=True).add_legend()
# sns.FacetGrid(df4, hue = 'activity', height = 4).map(sns.distplot, 'c13', kde=True).add_legend()
# sns.FacetGrid(df5, hue = 'activity', height = 4).map(sns.distplot, 'c13', kde=True).add_legend()
# sns.FacetGrid(df6, hue = 'activity', height = 4).map(sns.distplot, 'c13', kde=True).add_legend()
#
# plt.show()








# # RANDOM FOREST pentru PTBDB
# n = 200
# md = 20
# ms = 0.9
# rfc = RandomForestClassifier(n_estimators=n, max_depth=md, max_samples=ms)
# rfc.fit(Xtrain, ytrain)
# y_pred = rfc.predict(Xtest)
# precision, recall, f1_score, support = precision_recall_fscore_support(ytest, y_pred)
# class_accuracy = pd.DataFrame({
#     'Class 0': [precision[0], recall[0], f1_score[0]],
#     'Class 1': [precision[1], recall[1], f1_score[1]]
# }, index=['Precision', 'Recall', 'F1-score'])
# scores = cross_val_score(rfc, Xtest, ytest, cv=5)  # cv specifică numărul de fold-uri pentru cross-validation
#
# print("Parametrii: n_estimators=",n,", max_depth=",md,", max_samples=",ms)
# print(f"Class-wise Precision / Recall / F1-score: \n{class_accuracy}")
# print("Mean accuracy:", scores.mean())
# print("Std accuracy:", scores.std())




# RANDOM FOREST CU GRID pentru PTBDB

# # separați setul de date în variabile de caracteristici (X) și variabila de clasă (y)
# Xtrain = ptbdb_train.iloc[:, :-1]
# ytrain = ptbdb_train.iloc[:, -1]
# Xtest = ptbdb_test.iloc[:, :-1]
# ytest = ptbdb_test.iloc[:, -1]
#
# rfc = RandomForestClassifier()
# # print('valoarea n estimators implicita: ', rfc.n_estimators) # Afișează valoarea implicită
# param_grid = {
#     'n_estimators': [50, 100, 200], # numărul de arbori
#     'max_depth': [5, 10, 20],       # adâncimea maximă a unui arbore
#     'max_samples': [0.5, 0.7, 0.9]  # procentul din input folosit la antrenarea fiecărui arbore
# }
#
# grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)
# grid_search.fit(Xtrain, ytrain)
# best_rf = grid_search.best_estimator_
#
# y_pred = best_rf.predict(Xtest)
# accuracy = best_rf.score(Xtest, ytest)
# report = classification_report(ytest, y_pred)
# precision, recall, f1_score, support = precision_recall_fscore_support(ytest, y_pred)
#
# class_accuracy = pd.DataFrame({
#     'Class 0': [precision[0], recall[0], f1_score[0], support[0]],
#     'Class 1': [precision[1], recall[1], f1_score[1], support[1]]
# }, index=['Precision', 'Recall', 'F1-score', 'Support'])
#
# class_mean = class_accuracy.mean()
# class_std = class_accuracy.std()
#
# scores = cross_val_score(rfc, Xtest, ytest, cv=5)
#
# print(f"Accuracy: {accuracy}")
# print('Best parameters:', best_rf)
# print(f"Class-wise Precision / Recall / F1-score: \n{class_accuracy}")
# print("Mean accuracy:", scores.mean())
# print("Std accuracy:", scores.std())




# # GRADIENTBOOSTED TREES CU GRID pentru PTBDB
# # separați setul de date în variabile de caracteristici (X) și variabila de clasă (y)
# Xtrain = ptbdb_train.iloc[:, :-1]
# ytrain = ptbdb_train.iloc[:, -1]
# Xtest = ptbdb_test.iloc[:, :-1]
# ytest = ptbdb_test.iloc[:, -1]
#
# xgb_model = xgb.XGBClassifier()
#
# param_grid = {
#     'n_estimators': [100, 500, 1000],   # [100, 500, 1000]
#     'max_depth': [3, 5, 7],             # [3, 5, 7]
#     'learning_rate': [0.01, 0.1, 0.5],  # [0.01, 0.1, 0.5]
# }
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
# )
# grid_search.fit(Xtrain, ytrain)
# print('Best parameters:', grid_search.best_params_)
# print('Best cross-validation score:', grid_search.best_score_)
# best_xgb_model = xgb.XGBClassifier(
#     n_estimators=grid_search.best_params_['n_estimators'],
#     max_depth=grid_search.best_params_['max_depth'],
#     learning_rate=grid_search.best_params_['learning_rate'],
# )
#
# best_xgb_model.fit(Xtrain, ytrain)
#
# y_pred = best_xgb_model.predict(Xtest)
# print("Accuracy:", accuracy_score(ytest, y_pred))
# report = classification_report(ytest, y_pred)
# print(report)
# scores = cross_val_score(best_xgb_model, Xtrain, ytrain, cv=5)
# print("Mean accuracy: {:.2f}".format(scores.mean()))
# print("Standard deviation:", scores.std())




# # SVM CU GRID pentru PTBDB
# Xtrain = ptbdb_train.iloc[:, :-1]
# ytrain = ptbdb_train.iloc[:, -1]
# Xtest = ptbdb_test.iloc[:, :-1]
# ytest = ptbdb_test.iloc[:, -1]
#
# parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}
# svc = SVC()
# clf = GridSearchCV(svc, parameters)
#
# clf.fit(Xtrain, ytrain)
#
# best_params = clf.best_params_
# accuracy = clf.score(Xtest, ytest)
#
# y_pred = clf.predict(Xtest)
# report = classification_report(ytest, y_pred)
#
# scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
# mean_acc = scores.mean()
# std_acc = scores.std()
#
# print("Best parameters: ", best_params)
# print("Accuracy: ", accuracy)
# print("Classification report:\n", report)
# print("Mean accuracy: ", mean_acc)
# print("Std accuracy: ", std_acc)






# EXTRAGEREA ATRIBUTELOR PENTRU PTBDB
# pastrez ultima coloana
last_col_ptbdb_test = ptbdb_test.iloc[:, -1]

medii_ptbdb_test = ptbdb_test.iloc[:, :-1].mean(axis=1)
abatere_standard_ptbdb_test = ptbdb_test.iloc[:, :-1].std(axis=1)
abatere_medie_absoluta_ptbdb_test = ptbdb_test.iloc[:, :-1].apply(lambda x: np.mean(np.abs(x - x.mean())), axis=1)
valori_minime_ptbdb_test = ptbdb_test.iloc[:, :-1].min(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din ptbdb_test
valori_maxime_ptbdb_test = ptbdb_test.iloc[:, :-1].max(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din ptbdb_test
diferente_max_min_ptbdb_test = ptbdb_test.iloc[:, :-1].max(axis=1) - ptbdb_test.iloc[:, :-1].min(axis=1)
mediana_ptbdb_test = ptbdb_test.iloc[:, :-1].median(axis=1)
abaterea_mediana_absoluta_ptbdb_test = ptbdb_test.iloc[:, :-1].apply(lambda x: np.median(np.abs(x - np.median(x))), axis=1)
intervalul_intercuartil_ptbdb_test = ptbdb_test.iloc[:, :-1].apply(lambda x: np.subtract(*np.percentile(x, [75, 25])), axis=1)
numar_valori_negative_ptbdb_test = ptbdb_test.iloc[:, :-1].apply(lambda x: (x < 0).sum(), axis=1)
numar_valori_pozitive_ptbdb_test = ptbdb_test.iloc[:,:-1].apply(lambda x: (x > 0).sum(), axis=1)
numar_valori_peste_medie_ptbdb_test = pd.DataFrame((ptbdb_test.iloc[:, :-1] > ptbdb_test.iloc[:, :-1].mean(axis=1)[:, np.newaxis]).sum(axis=1))
# Extragem vârfurile pentru fiecare linie în parte
peaks = [find_peaks(row[:-1])[0] for index, row in ptbdb_test.iloc[:,:-1].iterrows()]
# Numărul de vârfuri pentru fiecare linie în parte
lista_numar_varfuri = [len(p) for p in peaks]
numar_varfuri_ptbdb_test = pd.Series(lista_numar_varfuri, index=ptbdb_test.index)

def calculate_energy(row):
    return np.mean(np.square(row))

energia_ptbdb_test = ptbdb_test.iloc[:, :-1].apply(calculate_energy, axis=1)
asimetrie_ptbdb_test = ptbdb_test.iloc[:, :-1].skew(axis=1)
curtoza_ptbdb_test = ptbdb_test.iloc[:, :-1].apply(kurtosis, axis=1)

dfs_atribute_ptbdb_test = [medii_ptbdb_test, abatere_standard_ptbdb_test, abatere_medie_absoluta_ptbdb_test,
                           valori_minime_ptbdb_test, valori_maxime_ptbdb_test, diferente_max_min_ptbdb_test,
                           mediana_ptbdb_test, abaterea_mediana_absoluta_ptbdb_test, intervalul_intercuartil_ptbdb_test,
                           numar_valori_negative_ptbdb_test, numar_valori_pozitive_ptbdb_test,
                           numar_valori_peste_medie_ptbdb_test, numar_varfuri_ptbdb_test, energia_ptbdb_test,
                           asimetrie_ptbdb_test, curtoza_ptbdb_test]

df_atribute_ptbdb_test = pd.concat(dfs_atribute_ptbdb_test, axis=1)
df_atribute_ptbdb_test = pd.concat([df_atribute_ptbdb_test, last_col_ptbdb_test], axis=1)
# print(df_atribute_ptbdb_test)

# matricea de corelație
# corr_matrix = df_atribute_ptbdb_test.corr()
# print(corr_matrix)


# pastrez ultima coloana
last_col_ptbdb_train = ptbdb_train.iloc[:, -1]

medii_ptbdb_train = ptbdb_train.iloc[:, :-1].mean(axis=1)
abatere_standard_ptbdb_train = ptbdb_train.iloc[:, :-1].std(axis=1)
abatere_medie_absoluta_ptbdb_train = ptbdb_train.iloc[:, :-1].apply(lambda x: np.mean(np.abs(x - x.mean())), axis=1)
valori_minime_ptbdb_train = ptbdb_train.iloc[:, :-1].min(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din ptbdb_train
valori_maxime_ptbdb_train = ptbdb_train.iloc[:, :-1].max(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din ptbdb_train
diferente_max_min_ptbdb_train = ptbdb_train.iloc[:, :-1].max(axis=1) - ptbdb_train.iloc[:, :-1].min(axis=1)
mediana_ptbdb_train = ptbdb_train.iloc[:, :-1].median(axis=1)
abaterea_mediana_absoluta_ptbdb_train = ptbdb_train.iloc[:, :-1].apply(lambda x: np.median(np.abs(x - np.median(x))), axis = 1)
intervalul_intercuartil_ptbdb_train = ptbdb_train.iloc[:, :-1].apply(lambda x: np.subtract(*np.percentile(x, [75, 25])), axis=1)
numar_valori_negative_ptbdb_train = ptbdb_train.iloc[:, :-1].apply(lambda x: (x < 0).sum(), axis=1)
numar_valori_pozitive_ptbdb_train = ptbdb_train.iloc[:,:-1].apply(lambda x: (x > 0).sum(), axis=1)
numar_valori_peste_medie_ptbdb_train = pd.DataFrame((ptbdb_train.iloc[:, :-1] > ptbdb_train.iloc[:, :-1].mean(axis=1)[:, np.newaxis]).sum(axis=1))
# Extragem vârfurile pentru fiecare linie în parte
peaks = [find_peaks(row[:-1])[0] for index, row in ptbdb_train.iloc[:,:-1].iterrows()]
# Numărul de vârfuri pentru fiecare linie în parte
lista_numar_varfuri = [len(p) for p in peaks]
numar_varfuri_ptbdb_train = pd.Series(lista_numar_varfuri, index=ptbdb_train.index)
def calculate_energy(row):
    return np.mean(np.square(row))
energia_ptbdb_train = ptbdb_train.iloc[:,:-1].apply(calculate_energy, axis=1)
asimetrie_ptbdb_train = ptbdb_train.iloc[:,:-1].skew(axis=1)
curtoza_ptbdb_train = ptbdb_train.iloc[:,:-1].apply(kurtosis, axis=1)

dfs_atribute_ptbdb_train = [medii_ptbdb_train, abatere_standard_ptbdb_train, abatere_medie_absoluta_ptbdb_train,
                            valori_minime_ptbdb_train, valori_maxime_ptbdb_train, diferente_max_min_ptbdb_train,
                            mediana_ptbdb_train, abaterea_mediana_absoluta_ptbdb_train,
                            intervalul_intercuartil_ptbdb_train, numar_valori_negative_ptbdb_train,
                            numar_valori_pozitive_ptbdb_train, numar_valori_peste_medie_ptbdb_train,
                            numar_varfuri_ptbdb_train, energia_ptbdb_train, asimetrie_ptbdb_train,
                            curtoza_ptbdb_train]

df_atribute_ptbdb_train = pd.concat(dfs_atribute_ptbdb_train, axis=1)
df_atribute_ptbdb_train = pd.concat([df_atribute_ptbdb_train, last_col_ptbdb_train], axis=1)
# print(df_atribute_ptbdb_train)

# matricea de corelație
# corr_matrix = df_atribute_ptbdb_train.corr()
# print(corr_matrix)




# SVM CU GRID pentru ATRIBUTE PTBDB
Xtrain = df_atribute_ptbdb_train.iloc[:, :-1]
ytrain = df_atribute_ptbdb_train.iloc[:, -1]
Xtest = df_atribute_ptbdb_test.iloc[:, :-1]
ytest = df_atribute_ptbdb_test.iloc[:, -1]

parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)

clf.fit(Xtrain, ytrain)

best_params = clf.best_params_
accuracy = clf.score(Xtest, ytest)

y_pred = clf.predict(Xtest)
report = classification_report(ytest, y_pred)

scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
mean_acc = scores.mean()
std_acc = scores.std()

print("Best parameters: ", best_params)
print("Accuracy: ", accuracy)
print("Classification report:\n", report)
print("Mean accuracy: ", mean_acc)
print("Std accuracy: ", std_acc)




# # EXTRAGEREA ATRIBUTELOR PENTRU MITBIH
# # pastrez ultima coloana
# last_col_mitbih_test = mitbih_test.iloc[:, -1]
#
# medii_mitbih_test = mitbih_test.iloc[:, :-1].mean(axis=1)
# abatere_standard_mitbih_test = mitbih_test.iloc[:, :-1].std(axis=1)
# abatere_medie_absoluta_mitbih_test = mitbih_test.iloc[:, :-1].apply(lambda x: np.mean(np.abs(x - x.mean())), axis=1)
# valori_minime_mitbih_test = mitbih_test.iloc[:, :-1].min(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din mitbih_test
# valori_maxime_mitbih_test = mitbih_test.iloc[:, :-1].max(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din mitbih_test
# diferente_max_min_mitbih_test = mitbih_test.iloc[:, :-1].max(axis=1) - mitbih_test.iloc[:, :-1].min(axis=1)
# mediana_mitbih_test = mitbih_test.iloc[:, :-1].median(axis=1)
# abaterea_mediana_absoluta_mitbih_test = mitbih_test.iloc[:, :-1].apply(lambda x: np.median(np.abs(x - np.median(x))), axis = 1)
# intervalul_intercuartil_mitbih_test = mitbih_test.iloc[:, :-1].apply(lambda x: np.subtract(*np.percentile(x, [75, 25])), axis=1)
# numar_valori_negative_mitbih_test = mitbih_test.iloc[:, :-1].apply(lambda x: (x < 0).sum(), axis=1)
# numar_valori_pozitive_mitbih_test = mitbih_test.iloc[:,:-1].apply(lambda x: (x > 0).sum(), axis=1)
# numar_valori_peste_medie_mitbih_test = pd.DataFrame((mitbih_test.iloc[:, :-1] > mitbih_test.iloc[:, :-1].mean(axis=1)[:, np.newaxis]).sum(axis=1))
# # Extragem vârfurile pentru fiecare linie în parte
# peaks = [find_peaks(row[:-1])[0] for index, row in mitbih_test.iloc[:,:-1].iterrows()]
# # Numărul de vârfuri pentru fiecare linie în parte
# lista_numar_varfuri = [len(p) for p in peaks]
# numar_varfuri_mitbih_test = pd.Series(lista_numar_varfuri, index=mitbih_test.index)
# def calculate_energy(row):
#     return np.mean(np.square(row))
# energia_mitbih_test = mitbih_test.iloc[:,:-1].apply(calculate_energy, axis=1)
# asimetrie_mitbih_test = mitbih_test.iloc[:,:-1].skew(axis=1)
# curtoza_mitbih_test = mitbih_test.iloc[:,:-1].apply(kurtosis, axis=1)
#
# dfs_atribute_mitbih_test = [medii_mitbih_test, abatere_standard_mitbih_test, abatere_medie_absoluta_mitbih_test, valori_minime_mitbih_test,
#                        valori_maxime_mitbih_test, diferente_max_min_mitbih_test, mediana_mitbih_test, abaterea_mediana_absoluta_mitbih_test,
#                        intervalul_intercuartil_mitbih_test, numar_valori_negative_mitbih_test, numar_valori_pozitive_mitbih_test,
#                        numar_valori_peste_medie_mitbih_test, numar_varfuri_mitbih_test, energia_mitbih_test, asimetrie_mitbih_test,
#                        curtoza_mitbih_test]
#
# df_atribute_mitbih_test = pd.concat(dfs_atribute_mitbih_test, axis=1)
# df_atribute_mitbih_test = pd.concat([df_atribute_mitbih_test, last_col_mitbih_test], axis=1)
# # print(df_atribute_mitbih_test)
#
#
#
# # pastrez ultima coloana
# last_col_mitbih_train = mitbih_train.iloc[:, -1]
#
# medii_mitbih_train = mitbih_train.iloc[:, :-1].mean(axis=1)
# abatere_standard_mitbih_train = mitbih_train.iloc[:, :-1].std(axis=1)
# abatere_medie_absoluta_mitbih_train = mitbih_train.iloc[:, :-1].apply(lambda x: np.mean(np.abs(x - x.mean())), axis=1)
# valori_minime_mitbih_train = mitbih_train.iloc[:, :-1].min(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din mitbih_train
# valori_maxime_mitbih_train = mitbih_train.iloc[:, :-1].max(axis=1) # serie pandas cu indicele corespunzător fiecărei linii din mitbih_train
# diferente_max_min_mitbih_train = mitbih_train.iloc[:, :-1].max(axis=1) - mitbih_train.iloc[:, :-1].min(axis=1)
# mediana_mitbih_train = mitbih_train.iloc[:, :-1].median(axis=1)
# abaterea_mediana_absoluta_mitbih_train = mitbih_train.iloc[:, :-1].apply(lambda x: np.median(np.abs(x - np.median(x))), axis = 1)
# intervalul_intercuartil_mitbih_train = mitbih_train.iloc[:, :-1].apply(lambda x: np.subtract(*np.percentile(x, [75, 25])), axis=1)
# numar_valori_negative_mitbih_train = mitbih_train.iloc[:, :-1].apply(lambda x: (x < 0).sum(), axis=1)
# numar_valori_pozitive_mitbih_train = mitbih_train.iloc[:,:-1].apply(lambda x: (x > 0).sum(), axis=1)
# numar_valori_peste_medie_mitbih_train = pd.DataFrame((mitbih_train.iloc[:, :-1] > mitbih_train.iloc[:, :-1].mean(axis=1)[:, np.newaxis]).sum(axis=1))
# # Extragem vârfurile pentru fiecare linie în parte
# peaks = [find_peaks(row[:-1])[0] for index, row in mitbih_train.iloc[:,:-1].iterrows()]
# # Numărul de vârfuri pentru fiecare linie în parte
# lista_numar_varfuri = [len(p) for p in peaks]
# numar_varfuri_mitbih_train = pd.Series(lista_numar_varfuri, index=mitbih_train.index)
# def calculate_energy(row):
#     return np.mean(np.square(row))
# energia_mitbih_train = mitbih_train.iloc[:,:-1].apply(calculate_energy, axis=1)
# asimetrie_mitbih_train = mitbih_train.iloc[:,:-1].skew(axis=1)
# curtoza_mitbih_train = mitbih_train.iloc[:,:-1].apply(kurtosis, axis=1)
#
# dfs_atribute_mitbih_train = [medii_mitbih_train, abatere_standard_mitbih_train, abatere_medie_absoluta_mitbih_train, valori_minime_mitbih_train,
#                        valori_maxime_mitbih_train, diferente_max_min_mitbih_train, mediana_mitbih_train, abaterea_mediana_absoluta_mitbih_train,
#                        intervalul_intercuartil_mitbih_train, numar_valori_negative_mitbih_train, numar_valori_pozitive_mitbih_train,
#                        numar_valori_peste_medie_mitbih_train, numar_varfuri_mitbih_train, energia_mitbih_train, asimetrie_mitbih_train,
#                        curtoza_mitbih_train]
#
# df_atribute_mitbih_train = pd.concat(dfs_atribute_mitbih_train, axis=1)
# df_atribute_mitbih_train = pd.concat([df_atribute_mitbih_train, last_col_mitbih_train], axis=1)
# # print(df_atribute_mitbih_train)
