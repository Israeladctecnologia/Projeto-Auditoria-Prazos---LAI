# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# MODELO: ÁRVORE DE DECISÃO (CLASSIFICAÇÃO) - BASE 2023 A 2025 - VERSÃO 3
#
# Antes de rodar, execute o Tratamento_LAI_2023_2025.py (gera o DadosAbertos_LAI_2023_2025_tratado.csv).
#
# DIFERENÇAS EM RELAÇÃO AO Classification_Decision_Tree_2024_v2.py:
#   1. DADOS: três anos (2023, 2024 e 2025, ~379 mil pedidos federais) em vez de só 2024 (~117 mil).
#   2. HISTÓRICO DO ÓRGÃO EM JANELA DE 12 MESES: a v2 acumulava todo o histórico desde o início da base.
#      Com três anos, um atraso de 2023 pesaria igual a um de ontem. Agora HistAtraso12m considera só as
#      respostas do órgão nos 12 meses anteriores ao registro do pedido. Testamos as duas formas e a
#      janela de 12 meses foi melhor no teste de 2025 (AUC 0,916 contra 0,905).
#   3. CATEGORIAS RARAS: limite de 1.500 pedidos (equivale aos 500 por ano da v2, agora com três anos).
#   4. LIMIAR: 5% (na v2 era 3%). Com este limiar o modelo atinge recall acima de 80% nos dois testes.
#   5. NOVA SEÇÃO 8C - TESTE TEMPORAL: treina com 2023-2024 e testa em 2025, simulando o uso real
#      (aprender com o passado e prever pedidos novos). Também compara com treinar só com 2024,
#      para medir se mais anos de dados ajudam.
#   6. RANKING DE ÓRGÃOS POR ANO (seção 10B): mostra se a taxa de atraso dos órgãos melhorou ou piorou.
#   O restante (alvo, one-hot, divisão 80/20, árvore, métricas, pedido novo) segue o mesmo padrão.
# ==============================================================================

# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
# ==============================================================================

# Path ajuda a localizar o arquivo CSV na mesma pasta deste script, não importa de onde você execute.
from pathlib import Path

# NumPy: usado aqui para contas com datas e para trabalhar com vetores de números.
import numpy as np

# Importa o Pandas para criação, manipulação e estruturação de dados em tabelas (DataFrames).
import pandas as pd

# Importa a função para dividir os dados em conjuntos distintos de treinamento e teste.
from sklearn.model_selection import train_test_split

# Importa a classe do algoritmo de Árvore de Decisão, a função para extrair suas regras em texto
# e a função para desenhar a árvore.
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# Importa funções para calcular métricas de desempenho: acurácia, precisão, recall, F1, AUC e matriz de confusão.
# A AUC mede o quão bem o modelo ORDENA os pedidos do menos para o mais arriscado (0,5 = sorteio; 1,0 = perfeito).
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix, roc_auc_score)

# Importa o módulo PyPlot do Matplotlib para geração e customização de gráficos e figuras.
import matplotlib.pyplot as plt

# Importa a classe do scikit-learn para formatar e desenhar graficamente a matriz de confusão.
from sklearn.metrics import ConfusionMatrixDisplay


# ==============================================================================
# 2. CARREGAMENTO DOS DADOS E CONVERSÃO DAS DATAS
# ==============================================================================

pasta = Path(__file__).parent
caminho_csv = pasta / "DadosAbertos_LAI_2023_2025_tratado.csv"

# Lê o CSV tratado (UTF-8 com marca do Excel, por isso 'utf-8-sig').
df = pd.read_csv(caminho_csv, sep=';', encoding='utf-8-sig')

# Converte as colunas de data de texto ("02/01/2024") para o tipo data do Pandas.
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y', errors='coerce')

print("=== Dados carregados ===")
print(f"Registros: {len(df)} | Colunas: {df.shape[1]}")
print(df['AnoArquivo'].value_counts().sort_index().rename('Pedidos por ano').to_string(), "\n")


# ==============================================================================
# 3. CRIAÇÃO DA VARIÁVEL-ALVO (TARGET)
# ==============================================================================

# Cria a coluna DentroDoPrazo: 1 se a resposta saiu até o prazo (DataResposta <= PrazoAtendimento), 0 se saiu depois.
df['DentroDoPrazo'] = (df['DataResposta'] <= df['PrazoAtendimento']).astype(int)

# Coluna auxiliar: 1 = ATRASOU (o inverso do alvo). Facilita as contas de taxa de atraso mais abaixo.
df['Atrasou'] = 1 - df['DentroDoPrazo']

print("=== Distribuição da variável-alvo ===")
print(df['DentroDoPrazo'].value_counts().rename({1: 'Dentro do prazo (1)', 0: 'Fora do prazo (0)'}))
print("Taxa de atraso por ano:", df.groupby('AnoArquivo')['Atrasou'].mean().round(4).to_dict(), "\n")


# ==============================================================================
# 4. SELEÇÃO E TRANSFORMAÇÃO DAS VARIÁVEIS DE ENTRADA (FEATURES)
# ==============================================================================

# Variável: quantos dias de prazo o órgão tinha (PrazoAtendimento - DataRegistro).
# Atenção: nos pedidos prorrogados o prazo já inclui a prorrogação (ex.: 20 dias viram 30+).
df['DiasDePrazo'] = (df['PrazoAtendimento'] - df['DataRegistro']).dt.days

# Variável: mês do registro, para capturar épocas do ano mais difíceis.
df['MesRegistro'] = df['DataRegistro'].dt.month

# Variáveis de histórico do órgão (para cada órgão, olhamos os pedidos dele em ordem de data):
# - HistAtraso12m (MUDOU): entre as respostas que o órgão deu nos 12 MESES ANTERIORES ao registro do pedido,
#   que fração saiu fora do prazo. Só usa informação que já existia na data do registro (sem "espiar o futuro").
# - Carga30: quantos pedidos o órgão recebeu nos 30 dias anteriores a este (sobrecarga de trabalho).
# O np.searchsorted descobre "quantos itens de uma lista ordenada são menores que X"; com ele contamos
# quantas respostas (e quantos atrasos) caem dentro de cada janela de datas.
JANELA_DIAS = 365
taxa_geral = df['Atrasou'].mean()                 # usada quando o órgão não tem nenhuma resposta na janela
df['HistAtraso12m'] = taxa_geral
df['Carga30'] = 0
for orgao, grupo in df.groupby('OrgaoDestinatario'):
    # Datas como números inteiros (dias), para poder contar com searchsorted.
    registro = grupo['DataRegistro'].values.astype('datetime64[D]').astype(int)
    resposta = grupo['DataResposta'].values.astype('datetime64[D]').astype(int)
    atraso = grupo['Atrasou'].values

    # Ordena as respostas do órgão por data e monta a soma acumulada de atrasos.
    # O 0 no início facilita a conta "atrasos entre a posição i e a posição j" = acum[j] - acum[i].
    ordem = np.argsort(resposta)
    resposta_ord = resposta[ordem]
    atraso_acum = np.concatenate([[0], np.cumsum(atraso[ordem])])

    # Posição das respostas anteriores ao registro (fim da janela) e anteriores a 12 meses antes (início da janela).
    fim = np.searchsorted(resposta_ord, registro, side='left')
    inicio = np.searchsorted(resposta_ord, registro - JANELA_DIAS, side='left')
    respostas_na_janela = fim - inicio
    atrasos_na_janela = atraso_acum[fim] - atraso_acum[inicio]
    df.loc[grupo.index, 'HistAtraso12m'] = np.where(respostas_na_janela > 0,
                                                    atrasos_na_janela / np.maximum(respostas_na_janela, 1),
                                                    taxa_geral)

    # Carga30: pedidos do órgão registrados entre (data - 30 dias) e a data atual.
    registro_ord = np.sort(registro)
    df.loc[grupo.index, 'Carga30'] = (np.searchsorted(registro_ord, registro, side='left')
                                      - np.searchsorted(registro_ord, registro - 30, side='left'))

# Escolha das variáveis de entrada (mesmas da v2, com o histórico agora em janela de 12 meses):
# - FICAM: OrgaoDestinatario, AssuntoPedido, FoiProrrogado, FoiReencaminhado, OrigemSolicitacao,
#          DiasDePrazo, MesRegistro, HistAtraso12m, Carga30.
# - FORA (identificadores): IdPedido e IdSolicitante. São apenas "números de série"; não explicam atraso.
# - FORA (vazamento de informação / data leakage): DataResposta, Decisao e EspecificacaoDecisao só existem
#   DEPOIS que o pedido foi respondido. DataResposta, além disso, é usada para criar o alvo.
# - FORA (pouca informação): Esfera, Situacao e FormaResposta são praticamente constantes após o tratamento.
# - FORA: AnoArquivo. O ano serve para separar treino e teste no teste temporal; como variável, o modelo
#   aprenderia "2025" e não teria como usar isso para um pedido de 2026.
colunas_entrada = ['OrgaoDestinatario', 'AssuntoPedido', 'FoiProrrogado', 'FoiReencaminhado',
                   'OrigemSolicitacao', 'DiasDePrazo', 'MesRegistro', 'HistAtraso12m', 'Carga30']
X = df[colunas_entrada].copy()

# Seleciona a coluna alvo (1 = dentro do prazo, 0 = fora do prazo) e armazena no vetor y.
y = df['DentroDoPrazo']

# Agrupa categorias raras em "Outros". Limite de 1.500 pedidos nos três anos (equivale aos 500 por ano da v2).
LIMITE_RARO = 1500
for coluna in ['OrgaoDestinatario', 'AssuntoPedido']:
    frequencia = X[coluna].value_counts()
    raras = frequencia[frequencia < LIMITE_RARO].index
    X[coluna] = X[coluna].where(~X[coluna].isin(raras), 'Outros')

# ONE-HOT ENCODING (pd.get_dummies): cada categoria vira uma coluna 0/1. drop_first=True remove a coluna redundante.
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("=== Variáveis de entrada após a transformação ===")
print(f"{X.shape[1]} colunas numéricas\n")


# ==============================================================================
# 5. DIVISÃO DOS DADOS EM TREINO E TESTE (80/20 ALEATÓRIO, COMO NAS VERSÕES ANTERIORES)
# ==============================================================================

# Divide X e y em treino (80%) e teste (20%), com a mesma proporção de atrasos nos dois (stratify=y).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Treino: {len(X_train)} pedidos ({(y_train == 0).sum()} fora do prazo)")
print(f"Teste:  {len(X_test)} pedidos ({(y_test == 0).sum()} fora do prazo)\n")


# ==============================================================================
# 6. INSTANCIAÇÃO E TREINAMENTO DO MODELO
# ==============================================================================

# Árvore de Decisão (mesma configuração da v2):
# - max_depth=4: limita a profundidade para evitar sobreajuste e manter as regras legíveis.
# - min_samples_leaf=100: cada regra final precisa ter pelo menos 100 pedidos (evita regras feitas de acaso).
# - SEM class_weight: a probabilidade de cada pedido fica próxima da frequência real de atraso.
modelo = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
modelo.fit(X_train, y_train)

# Nomes das classes na ordem do modelo (0 primeiro, depois 1), usados nos relatórios e gráficos.
nomes_classes = ['Fora do prazo (0)', 'Dentro do prazo (1)']

# Probabilidade de cada pedido ser atendido FORA do prazo (coluna 0 do predict_proba).
prob_fora_teste = modelo.predict_proba(X_test)[:, 0]
prob_fora_treino = modelo.predict_proba(X_train)[:, 0]


# ==============================================================================
# 7. REGRA DE DECISÃO: LIMIAR DE CORTE
# ==============================================================================

# Se a probabilidade de atraso for maior ou igual ao LIMIAR, o pedido gera um alerta ("fora do prazo").
# - Limiar BAIXO  -> mais alertas, pega mais atrasos (recall alto), mas muitos alarmes falsos.
# - Limiar ALTO   -> poucos alertas, mais certeiros, mas deixa passar mais atrasos.
# Escolhemos 5% porque, com ele, o modelo pega mais de 80% dos atrasos alertando cerca de 15% dos pedidos
# (veja a tabela da seção 8B). A taxa média de atraso é ~2,4%, então 5% já é "o dobro do risco médio".
LIMIAR = 0.05

# Converte probabilidades em classes: 0 (alerta de atraso) se prob_fora >= LIMIAR, senão 1 (dentro do prazo).
y_pred_train = np.where(prob_fora_treino >= LIMIAR, 0, 1)
y_pred = np.where(prob_fora_teste >= LIMIAR, 0, 1)


# ==============================================================================
# 8. AVALIAÇÃO DO MODELO NO TREINO E NO TESTE
# ==============================================================================

print("=== Acurácia no Treino ===")
print(f"Acurácia: {accuracy_score(y_train, y_pred_train):.2%}\n")

# Comparar treino e teste ajuda a perceber overfitting: se o treino for MUITO melhor que o teste, o modelo decorou.
print("=== Relatório de Classificação (Treino) ===")
print(classification_report(y_train, y_pred_train, target_names=nomes_classes, zero_division=0))

# Métricas no teste, com "fora do prazo" (pos_label=0) como classe de interesse.
taxa_base = (y_test == 0).mean()
precisao_teste = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
print(f"=== Métricas no Teste 80/20 (limiar = {LIMIAR:.0%}; classe de interesse: FORA do prazo) ===")
print(f"Acurácia  : {accuracy_score(y_test, y_pred):.2%}")
print(f"Precisão  : {precisao_teste:.2%}")
print(f"Recall    : {recall_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
print(f"F1-score  : {f1_score(y_test, y_pred, pos_label=0, zero_division=0):.2%}")
print(f"AUC       : {roc_auc_score(y_test == 0, prob_fora_teste):.3f}  (0,5 = sorteio; 1,0 = perfeito)")
print(f"Alertados : {(y_pred == 0).mean():.1%} dos pedidos")
# LIFT: quantas vezes a precisão do modelo é maior que a taxa de atraso (que seria a precisão de um sorteio).
print(f"Lift      : {precisao_teste / taxa_base:.1f}x melhor que sortear\n")

# Significado de cada métrica neste problema:
# - ACURÁCIA: % de pedidos classificados corretamente. Com ~97,6% de pedidos no prazo, dizer
#   "ninguém atrasa" já dá ~97,6% de acurácia sem detectar nada. Por isso ela é a menos útil aqui.
# - PRECISÃO: de todos os pedidos ALERTADOS, quantos realmente atrasaram (mede os alarmes falsos).
# - RECALL (revocação): de todos os pedidos que REALMENTE atrasaram, quantos o modelo alertou.
#   É o parâmetro principal (perder um atraso custa mais que olhar um pedido a mais), MAS sempre junto
#   da % de pedidos alertados (alertar todos daria recall de 100% e não serviria para nada).
# - F1-SCORE: média equilibrada entre precisão e recall.
# - AUC: qualidade da ORDENAÇÃO por risco, sem depender do limiar escolhido.
# - LIFT: precisão do modelo dividida pela taxa de atraso; fácil de explicar a quem não é da área.

print("=== Relatório de Classificação (Teste 80/20) ===")
print(classification_report(y_test, y_pred, target_names=nomes_classes, zero_division=0))


# ==============================================================================
# 8B. TABELA: O PREÇO DE CADA RECALL (VARIANDO O LIMIAR)
# ==============================================================================

def tabela_limiares(prob_fora, y_real, titulo):
    """Mostra, para vários limiares, a % de alertados, o recall, a precisão e o lift."""
    atrasou_real = (y_real == 0).values
    print(f"=== Trade-off por limiar ({titulo}) ===")
    print(f"{'Limiar':>7} | {'% alertados':>11} | {'Recall':>7} | {'Precisão':>8} | {'Lift':>5}")
    for lim in [0.02, 0.03, 0.05, 0.08, 0.12, 0.20]:
        alertas = prob_fora >= lim
        acertos = (alertas & atrasou_real).sum()
        recall_lim = acertos / atrasou_real.sum()
        precisao_lim = acertos / alertas.sum() if alertas.sum() > 0 else 0
        print(f"{lim:>7.0%} | {alertas.mean():>11.1%} | {recall_lim:>7.1%} | {precisao_lim:>8.1%} | "
              f"{precisao_lim / atrasou_real.mean():>5.1f}")
    print()


tabela_limiares(prob_fora_teste, y_test, "teste 80/20")


# ==============================================================================
# 8C. TESTE TEMPORAL: APRENDE COM 2023-2024 E PREVÊ 2025
# ==============================================================================

# É o teste mais próximo do uso real: o modelo só conhece o passado e precisa prever pedidos novos.
# Treinamos dois modelos e testamos os dois nos MESMOS pedidos de 2025:
#   (a) treinado só com 2024 -> mostra o que a v2 conseguiria em 2025;
#   (b) treinado com 2023 e 2024 -> mostra se ter mais anos de dados ajuda.
ano = df['AnoArquivo']
teste_2025 = ano == 2025
print(f"=== Teste temporal: {teste_2025.sum()} pedidos de 2025 ({(y[teste_2025] == 0).mean():.2%} atrasos) ===")
modelo_temporal = None
for nome, treino in [("treino só 2024", ano == 2024), ("treino 2023 + 2024", ano <= 2024)]:
    m = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=42)
    m.fit(X[treino], y[treino])
    prob = m.predict_proba(X[teste_2025])[:, 0]
    alerta = prob >= LIMIAR
    real = (y[teste_2025] == 0).values
    print(f"{nome:>20} ({treino.sum()} pedidos): AUC {roc_auc_score(real, prob):.3f} | "
          f"alertados {alerta.mean():.1%} | recall {(alerta & real).sum() / real.sum():.1%} | "
          f"precisão {(alerta & real).sum() / max(alerta.sum(), 1):.1%}")
    modelo_temporal, prob_temporal = m, prob
print()

# Tabela de limiares do modelo (b), o que seria usado na prática para prever pedidos de 2026.
tabela_limiares(prob_temporal, y[teste_2025], "teste temporal 2025, treino 2023+2024")


# ==============================================================================
# 9. MATRIZ DE CONFUSÃO E REGRAS APRENDIDAS
# ==============================================================================

# Matriz de confusão do teste 80/20 (linhas = realidade, colunas = previsão), na ordem 0 = Fora, 1 = Dentro:
#   [[ acertou "fora"            , previu "dentro" mas era "fora" (ATRASO NÃO DETECTADO) ],
#    [ previu "fora" mas era "dentro" (ALARME FALSO) , acertou "dentro"                   ]]
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("=== Matriz de Confusão (Teste 80/20) ===")
print(cm, "\n")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_classes)
disp.plot(cmap='Blues', values_format='d')
plt.xlabel("Previsão do modelo")     # eixos em português (o padrão do scikit-learn é em inglês)
plt.ylabel("Realidade")
plt.title(f"Matriz de Confusão - Teste 80/20 (limiar {LIMIAR:.0%}) - v3")
plt.savefig(pasta / "matriz_confusao_v3.png", dpi=150, bbox_inches='tight')
plt.show()

# Converte a estrutura lógica interna da árvore em texto estruturado por recuos (if/else).
print("=== Regras Aprendidas pelo Modelo ===")
print(export_text(modelo, feature_names=list(X.columns)))


# ==============================================================================
# 10. IMPORTÂNCIA DAS VARIÁVEIS E VISUALIZAÇÃO DA ÁRVORE
# ==============================================================================

# Importância: quanto cada variável ajudou a árvore a separar pedidos dentro e fora do prazo (soma 1).
importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
print("=== Importância das Variáveis (top 10) ===")
print(importancias.head(10).round(3).to_string(), "\n")

usadas = importancias[importancias > 0].sort_values()
plt.figure(figsize=(9, 5))
plt.barh(usadas.index, usadas.values, color='steelblue')
plt.xlabel("Importância")
plt.title("Importância das Variáveis - Fatores que mais influenciam o atraso (2023-2025)")
plt.tight_layout()
plt.savefig(pasta / "importancia_variaveis_v3.png", dpi=150, bbox_inches='tight')   # 'tight' evita cortar o título
plt.show()

# Desenha a árvore: cada caixa mostra a regra, a quantidade de pedidos e a classe predominante.
plt.figure(figsize=(34, 11))   # figura larga para as 16 folhas não se sobreporem
plot_tree(modelo, feature_names=list(X.columns), class_names=nomes_classes,
          filled=True, rounded=True, fontsize=8)
plt.title("Árvore de Decisão - Prazo de Atendimento (LAI 2023-2025) - v3")
plt.savefig(pasta / "arvore_decisao_v3.png", dpi=150, bbox_inches='tight')
plt.show()

# Versão RESUMIDA da árvore, só com os 2 primeiros níveis (max_depth=2 vale só para o DESENHO; o modelo
# continua com 4 níveis). A árvore completa tem 16 folhas e fica difícil de ler; o resumo mostra as perguntas
# mais importantes. proportion=True mostra porcentagens em vez de contagens: em "value = [0.024, 0.976]",
# 0.024 é a fração de pedidos atrasados naquele grupo.
plt.figure(figsize=(16, 7))
plot_tree(modelo, feature_names=list(X.columns), class_names=nomes_classes, max_depth=2,
          filled=True, rounded=True, fontsize=10, proportion=True)
plt.title("Árvore de Decisão v3 - resumo dos 2 primeiros níveis")
plt.savefig(pasta / "arvore_decisao_v3_resumo.png", dpi=150, bbox_inches='tight')
plt.show()


# ==============================================================================
# 10B. RANKING DE ÓRGÃOS COM MAIOR TAXA DE ATRASO, ANO A ANO (APOIO À AUDITORIA)
# ==============================================================================

# Tabela com uma linha por órgão: pedidos e atrasos nos três anos, taxa geral e taxa de cada ano.
# Só entram órgãos com pelo menos 90 pedidos nos três anos (média de 30 por ano), para a taxa não enganar.
ranking = df.groupby('OrgaoDestinatario')['Atrasou'].agg(Pedidos='size', Atrasos='sum', TaxaGeral='mean')
taxa_por_ano = df.pivot_table(index='OrgaoDestinatario', columns='AnoArquivo', values='Atrasou', aggfunc='mean')
taxa_por_ano.columns = [f'Taxa{ano_col}' for ano_col in taxa_por_ano.columns]
ranking = ranking.join(taxa_por_ano)
ranking = ranking[ranking['Pedidos'] >= 90].sort_values('TaxaGeral', ascending=False).round(3)
ranking.to_csv(pasta / "orgaos_taxa_atraso_2023_2025.csv", sep=';', encoding='utf-8-sig')
print("=== 10 órgãos com maior taxa de atraso em 2023-2025 (mín. 90 pedidos) ===")
print(ranking.head(10).to_string(), "\n")


# ==============================================================================
# 11. PREVISÃO PARA UM PEDIDO NOVO (EXEMPLO)
# ==============================================================================

# Usamos o modelo do teste temporal (treinado com 2023 e 2024), como seria feito para um pedido de 2026.
# O histórico do pedido novo vem dos dados mais recentes do órgão (2025): taxa de atraso e carga média.
orgao_exemplo = df['OrgaoDestinatario'].mode()[0]
recente = df[(df['OrgaoDestinatario'] == orgao_exemplo) & (ano == 2025)]

pedido_novo = pd.DataFrame([{
    'OrgaoDestinatario': orgao_exemplo,
    'AssuntoPedido': df['AssuntoPedido'].mode()[0],           # assunto mais frequente da base
    'FoiProrrogado': 'Sim',
    'FoiReencaminhado': 'Não',
    'OrigemSolicitacao': 'Internet',
    'DiasDePrazo': 30,
    'MesRegistro': 6,
    'HistAtraso12m': recente['Atrasou'].mean(),
    'Carga30': recente['Carga30'].mean(),
}])

# Aplica as MESMAS transformações do treino: one-hot e alinhamento das colunas com as do modelo
# (reindex preenche com 0 as colunas que não aparecem neste único pedido).
# Obs.: para um órgão/assunto raro, troque o valor por 'Outros', como foi feito no treino.
pedido_novo_cod = pd.get_dummies(pedido_novo, drop_first=False, dtype=int)
pedido_novo_cod = pedido_novo_cod.reindex(columns=X.columns, fill_value=0)

# predict_proba devolve a probabilidade de cada classe, na ordem [fora do prazo, dentro do prazo].
prob_fora, prob_dentro = modelo_temporal.predict_proba(pedido_novo_cod)[0]
print("=== Previsão para um pedido novo (exemplo) ===")
print(pedido_novo.round(3).to_string(index=False))
print(f"Probabilidade de ser atendido DENTRO do prazo: {prob_dentro:.1%}")
print(f"Probabilidade de ser atendido FORA do prazo:   {prob_fora:.1%}")
print(f"Gera alerta (limiar {LIMIAR:.0%})? {'SIM' if prob_fora >= LIMIAR else 'NÃO'}\n")


# ==============================================================================
# 12. COMO INTERPRETAR OS RESULTADOS E LIMITAÇÕES
# ==============================================================================
# COMO INTERPRETAR
# - O teste temporal (8C) é o mais importante: mostra o desempenho em pedidos de um ano que o modelo não viu.
# - Use as tabelas de limiar (8B/8C) para escolher o equilíbrio entre recall e quantidade de alertas que a
#   equipe consegue verificar.
# - As variáveis de maior importância e as regras impressas mostram QUAIS fatores a árvore associou ao atraso.
#   Isso indica associação, não prova de causa.
#
# LIMITAÇÕES
# 1. DESBALANCEAMENTO: só ~2,4% dos pedidos atrasaram. Com recall alto, a maioria dos alertas é falso alarme.
# 2. VARIÁVEIS LIMITADAS: o modelo não conhece a complexidade real do pedido, a equipe do órgão ou o texto do pedido.
# 3. FoiProrrogado e DiasDePrazo só são conhecidos depois que o órgão pede mais tempo (o prazo já inclui a
#    prorrogação). Logo, o modelo serve melhor para reavaliar pedidos em andamento do que para prever no registro.
# 4. INÍCIO DE 2023: não temos os dados de 2022, então nos primeiros meses de 2023 o HistAtraso12m usa uma janela
#    incompleta (e a taxa geral quando ainda não há nenhuma resposta).
# 5. SÓ FEDERAL: estados, municípios e outros poderes podem ter padrões diferentes.
# 6. MUDANÇAS NO TEMPO: a taxa de atraso subiu de 2,1% (2023) para 2,6% (2025). Se o comportamento dos órgãos
#    mudar, o modelo precisa ser retreinado periodicamente com os dados mais recentes.
