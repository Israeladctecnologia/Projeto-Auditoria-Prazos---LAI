# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# TESTE TEMPORAL: TREINA NO PASSADO E TESTA NO FUTURO
#
# Por que este teste? Na v2 os dados foram divididos ao acaso (80/20): pedidos de dezembro podem ter ficado no
# treino e pedidos de janeiro no teste. Na vida real o modelo só conhece o PASSADO e precisa prever o FUTURO.
# Aqui treinamos com pedidos registrados de jan a set e testamos com os de out a dez, que o modelo nunca viu.
# Se o desempenho se mantiver, é um sinal mais forte de que o modelo é útil de verdade.
#
# Este script reaproveita as mesmas variáveis e configurações do Classification_Decision_Tree_2024_v2.py.
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------
# 1. CARREGAMENTO, ALVO E VARIÁVEIS (igual à v2)
# ------------------------------------------------------------------------------
df = pd.read_csv(Path(__file__).parent / "DadosAbertos_LAI_2024_tratado.csv", sep=';', encoding='utf-8-sig')
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y', errors='coerce')

df['Atrasou'] = (df['DataResposta'] > df['PrazoAtendimento']).astype(int)   # 1 = atrasou
df['DiasDePrazo'] = (df['PrazoAtendimento'] - df['DataRegistro']).dt.days
df['MesRegistro'] = df['DataRegistro'].dt.month

# Histórico do órgão: taxa de atraso entre as respostas anteriores ao registro e pedidos nos 30 dias anteriores.
taxa_geral = df['Atrasou'].mean()
df['HistAtraso'] = taxa_geral
df['Carga30'] = 0
for orgao, grupo in df.groupby('OrgaoDestinatario'):
    registro = grupo['DataRegistro'].values.astype('datetime64[D]').astype(int)
    resposta = grupo['DataResposta'].values.astype('datetime64[D]').astype(int)
    atraso = grupo['Atrasou'].values
    ordem = np.argsort(resposta)
    atraso_acum = np.cumsum(atraso[ordem])
    qtd_antes = np.searchsorted(resposta[ordem], registro, side='left')
    atrasos_antes = np.where(qtd_antes > 0, atraso_acum[np.maximum(qtd_antes - 1, 0)], 0)
    df.loc[grupo.index, 'HistAtraso'] = np.where(qtd_antes > 0, atrasos_antes / np.maximum(qtd_antes, 1), taxa_geral)
    registro_ord = np.sort(registro)
    df.loc[grupo.index, 'Carga30'] = (np.searchsorted(registro_ord, registro, side='left')
                                      - np.searchsorted(registro_ord, registro - 30, side='left'))

colunas_entrada = ['OrgaoDestinatario', 'AssuntoPedido', 'FoiProrrogado', 'FoiReencaminhado',
                   'OrigemSolicitacao', 'DiasDePrazo', 'MesRegistro', 'HistAtraso', 'Carga30']
X = df[colunas_entrada].copy()
for coluna in ['OrgaoDestinatario', 'AssuntoPedido']:
    frequencia = X[coluna].value_counts()
    X[coluna] = X[coluna].where(~X[coluna].isin(frequencia[frequencia < 500].index), 'Outros')
X = pd.get_dummies(X, drop_first=True, dtype=int)
y = df['Atrasou']


# ------------------------------------------------------------------------------
# 2. FUNÇÃO QUE TREINA EM UM PERÍODO E AVALIA EM OUTRO
# ------------------------------------------------------------------------------
def avaliar(nome, mascara_treino, mascara_teste):
    """Treina com os pedidos da máscara de treino e avalia nos da máscara de teste (limiares em % de risco)."""
    modelo = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
    modelo.fit(X[mascara_treino], y[mascara_treino])
    risco = modelo.predict_proba(X[mascara_teste])[:, 1]       # probabilidade de ATRASO
    real = y[mascara_teste].values

    print(f"=== {nome} ===")
    print(f"Treino: {mascara_treino.sum()} pedidos ({y[mascara_treino].mean():.2%} atrasos) | "
          f"Teste: {mascara_teste.sum()} pedidos ({real.mean():.2%} atrasos)")
    print(f"AUC: {roc_auc_score(real, risco):.3f}")
    print(f"{'Limiar':>7} | {'% alertados':>11} | {'Recall':>7} | {'Precisão':>8} | {'Lift':>5}")
    for limiar in [0.02, 0.03, 0.05, 0.08]:
        alertas = risco >= limiar
        acertos = (alertas & (real == 1)).sum()
        recall = acertos / real.sum()
        precisao = acertos / alertas.sum() if alertas.sum() > 0 else 0
        print(f"{limiar:>7.0%} | {alertas.mean():>11.1%} | {recall:>7.1%} | {precisao:>8.1%} | {precisao / real.mean():>5.1f}")
    print()


# ------------------------------------------------------------------------------
# 3. OS TESTES
# ------------------------------------------------------------------------------
mes = df['MesRegistro']

# Teste temporal principal: treino jan-set, teste out-dez.
avaliar("TEMPORAL: treino jan-set -> teste out-dez", mes <= 9, mes >= 10)

# Variação: treino jan-ago, teste out-dez (deixa setembro de fora, como uma "folga" entre os períodos,
# porque pedidos de setembro podem ser respondidos só em outubro, já no período do teste).
avaliar("TEMPORAL com folga: treino jan-ago -> teste out-dez", mes <= 8, mes >= 10)

# Para comparar: resultado da divisão ALEATÓRIA da v2 (valores copiados da execução da v2).
# Se o teste temporal ficar bem pior que isto, a divisão aleatória estava otimista demais.
print("Referência (divisão aleatória da v2, limiar 3%): AUC 0,879 | recall 73,4% | precisão 12,6% | alertados 14,4%")
