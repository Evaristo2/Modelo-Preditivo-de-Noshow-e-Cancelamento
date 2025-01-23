import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# LENDO OS DADOS E TRATANDO-OS
df = pd.read_csv(df)

df['DATA_ID'] = pd.to_datetime(df['DATA_ID']) 

# PIVOTANDO A DURAÇÃO
df['DURACAO_GRUPO1'] = df['DURACAO'].apply(lambda x: 1 if 0 <= x <= 50 else 0)
df['DURACAO_GRUPO2'] = df['DURACAO'].apply(lambda x: 1 if x > 50 else 0)

# PIVOTANDO OS CANAIS
canal1 = ["LOJA - WALK IN", "MOBILE", "PARCERIAS", "PORTAL AGÊNCIA", "PORTAL EMPRESA", "WEBSERVICES (XML)", "WEBSITE MOVIDA"]
canal2 = ["MOTORISTA DE APLICATIVOS"]
canal3 = ["CALL CENTER", "GDS (AMADEUS)", "HOTSITE MENSAL FLEX", "LINK COLABORADOR", "MOVIDA CARGO", "SABRE"]
df['CANAL_GRUPO1'] = df['CANAL_NOME'].apply(lambda x: 1 if x in canal1 else 0)
df['CANAL_GRUPO2'] = df['CANAL_NOME'].apply(lambda x: 1 if x in canal2 else 0)
df['CANAL_GRUPO3'] = df['CANAL_NOME'].apply(lambda x: 1 if x in canal3 else 0)

# PIVOTANDO O STATUS
fechada = ["FECHADA"]
no_show = ["NO-SHOW"]
cancelado = ["CANCELADO"]
a_confirmar = ["À CONFIRMAR"]
df['FECHADA'] = df['STATUS_NOME'].apply(lambda x: 1 if x in fechada else 0)
df['NO_SHOW'] = df['STATUS_NOME'].apply(lambda x: 1 if x in no_show else 0)
df['CANCELADO'] = df['STATUS_NOME'].apply(lambda x: 1 if x in cancelado else 0)
df['A_CONFIRMAR'] = df['STATUS_NOME'].apply(lambda x: 1 if x in a_confirmar else 0)

# PIVOTANDO A MODALIDADE
mensal = ["MENSAL"]
eventual = ["DIARIA"]
df['MODALIDADE_MENSAL'] = df['RESERVA_MODALIDADE'].apply(lambda x: 1 if x in mensal else 0)
df['MODALIDADE_EVENTUAL'] = df['RESERVA_MODALIDADE'].apply(lambda x: 1 if x in eventual else 0)

# LÓGICA DO CANCELAMENTO DO CLIENTE
reservas_por_cliente = df.groupby('CLIENTE_ID').size().reset_index(name='TOTAL_RESERVAS')
cancelamentos_por_cliente = df[df['CANCELADO'] == 1].groupby('CLIENTE_ID').size().reset_index(name='TOTAL_CANCELAMENTOS')
cliente_cancelamento = pd.merge(reservas_por_cliente, cancelamentos_por_cliente, on='CLIENTE_ID', how='left')
cliente_cancelamento['TOTAL_CANCELAMENTOS'] = cliente_cancelamento['TOTAL_CANCELAMENTOS'].fillna(0)
cliente_cancelamento['CANCELAMENTO_PERCENT'] = (cliente_cancelamento['TOTAL_CANCELAMENTOS'] / cliente_cancelamento['TOTAL_RESERVAS']) * 100
df = pd.merge(df, cliente_cancelamento[['CLIENTE_ID', 'CANCELAMENTO_PERCENT']], on='CLIENTE_ID', how='left')

# SELEÇÃO DAS COLUNAS PARA O MODELO
df2 = df[[
          'CLIENTE_ID',
          'DATA_ID',
          'STATUS_NOME',
          'DURACAO_GRUPO1', 
          'DURACAO_GRUPO2', 
          'CANAL_GRUPO1',
          'CANAL_GRUPO2',
          'CANAL_GRUPO3',
          'MODALIDADE_MENSAL', 
          'MODALIDADE_EVENTUAL',
          'CANCELAMENTO_PERCENT',
          'CANCELADO']] 

if 'CANCELADO' not in df2.columns:
    raise KeyError("A coluna 'CANCELADO' não foi encontrada no DataFrame. Verifique se a coluna foi criada corretamente.")

# ADICIONANDO OS REGRESSORES (FEATURES)
features = [
            'MODALIDADE_MENSAL', 'MODALIDADE_EVENTUAL', 
            'CANAL_GRUPO1', 'CANAL_GRUPO2', 'CANAL_GRUPO3',
            'CANCELAMENTO_PERCENT']

# FEATURES (X) e TARGET (y)
X = df2[features]
y = df2['CANCELADO']

# BALANCEAMENTO DE DADOS - SMOTE
smote = SMOTE(random_state=7)
X_resampled, y_resampled = smote.fit_resample(X, y)

# TREINO E TESTE
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# HIPERPARÂMETROS
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# MODELO Random Forest Classifier
modelo_random_forest_cancelamento = RandomForestClassifier(random_state=7)

# ENCONTRAR OS MELHORES HIPERPARÂMETROS
random_search = RandomizedSearchCV(estimator=modelo_random_forest_cancelamento, param_distributions=param_dist, 
                                   n_iter=10, scoring='accuracy', cv=5, verbose=1, random_state=7, n_jobs=-1)
random_search.fit(X_train, y_train)

# SALVAR O MODELO TREINADO COM PICKLE
with open('predict_model2_cancelamento_sem_reserva_ID.pkl', 'wb') as arquivo:
    pickle.dump(random_search, arquivo)

# AVALIANDO O MODELO
y_pred = random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nMatriz de Confusão:")
print(conf_matrix)

# CALCULANDO AS MÉTRICAS PARA CANCELADO (1)
precision_cancelado = precision_score(y_test, y_pred, pos_label=1)
recall_cancelado = recall_score(y_test, y_pred, pos_label=1)
f1_cancelado = f1_score(y_test, y_pred, pos_label=1)

# CALCULANDO AS MÉTRICAS PARA não CANCELADO (0)
precision_non_cancelado = precision_score(y_test, y_pred, pos_label=0)
recall_non_cancelado = recall_score(y_test, y_pred, pos_label=0)
f1_non_cancelado = f1_score(y_test, y_pred, pos_label=0)

# PAINEL BONITO CONFUSION MATRIX
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

total = TN + FP + FN + TP

TN_percent = (TN / total) * 100
FP_percent = (FP / total) * 100
FN_percent = (FN / total) * 100
TP_percent = (TP / total) * 100

confusion_table = f"""
-----------------------------------------------------
| True Positive = {TP_percent:.2f}%  | False Negative = {FN_percent:.2f}% |
|-------------------------|-------------------------|
| False Positive = {FP_percent:.2f}%  |  True Negative = {TN_percent:.2f}% |
-----------------------------------------------------
"""

print('\nTabela de Confusão:', confusion_table)

# ACURÁCIA
print(f'Acurácia: {accuracy}')

# TRANSFORMANDO EM PORCENTAGEM
precision_cancelado_percent = precision_cancelado * 100
recall_cancelado_percent = recall_cancelado * 100
f1_cancelado_percent = f1_cancelado * 100

precision_non_cancelado_percent = precision_non_cancelado * 100
recall_non_cancelado_percent = recall_non_cancelado * 100
f1_non_cancelado_percent = f1_non_cancelado * 100

# TABELA EXECUTIVA
print("\nTabela de Métricas:")
print("-----------------------------------------------------")
print("|     CANCELADO     | Precisão | Recall  | F1-score |")
print("-----------------------------------------------------")
print(f"| Caso Negativo (0) | {precision_non_cancelado_percent:.2f}%   | {recall_non_cancelado_percent:.2f}%  | {f1_non_cancelado_percent:.2f}%  |")
print("-----------------------------------------------------")
print(f"| Caso Positivo (1) | {precision_cancelado_percent:.2f}%   | {recall_cancelado_percent:.2f}%  | {f1_cancelado_percent:.2f}%  |")
print("-----------------------------------------------------")

# PREVENDO E ADICIONANDO A COLUNA OCORRENCIA_CANCELADO NA BASE ORIGINAL
df['OCORRENCIA_CANCELADO'] = random_search.predict(df2[features])

# SALVANDO O DATAFRAME COM A COLUNA DE PREVISÕES
df.to_csv('db_com_Predict_CANCELADO2.csv', index=False)

print("\nPrimeiras linhas do DataFrame atualizado:")
print(df.head())
