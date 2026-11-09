# ==============================================================================
# PROJETO: ANÁLISE E AUDITORIA DE PRAZOS (Pedidos da LAI - Fala.BR)
# TRATAMENTO DOS DADOS (ETL) - ARQUIVOS DE 2023, 2024 E 2025 JUNTOS
#
# O que este script faz: junta os três arquivos brutos anuais em uma única base tratada
# ("DadosAbertos_LAI_2023_2025_tratado.csv"), usada pelo Classification_Decision_Tree_v3.py.
# Os arquivos originais NUNCA são alterados.
#
# DIFERENÇAS EM RELAÇÃO AO Tratamento_LAI_2024.py:
#   1. Lê TRÊS arquivos e os empilha (um embaixo do outro) em uma só tabela.
#   2. Os arquivos não vêm no mesmo formato: 2023 e 2025 estão em UTF-16; 2024 está em cp850 com aspas em
#      cada linha. Por isso a leitura virou uma função que recebe o formato de cada arquivo (seção 2).
#   3. O traço do nome dos órgãos vem diferente em cada ano: "–" (travessão) em 2023/2025 e "?" em 2024.
#      Padronizamos tudo para "-", senão o mesmo órgão apareceria com dois nomes diferentes.
#   4. Nova coluna AnoArquivo, para sabermos de qual ano veio cada pedido.
#   5. Novo filtro: remove pedidos sem PrazoAtendimento (há 1 caso em 2023).
#   O restante (colunas mantidas, valores ausentes, filtros de esfera, datas e duplicados) é igual.
# ==============================================================================

# ==============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
# ==============================================================================

# Path ajuda a localizar os arquivos na mesma pasta deste script, não importa de onde você execute.
from pathlib import Path

# Importa o Pandas para manipulação de dados em tabelas (DataFrames).
import pandas as pd


# ==============================================================================
# 2. EXTRAÇÃO: LEITURA DOS TRÊS ARQUIVOS BRUTOS
# ==============================================================================

pasta = Path(__file__).parent
arquivo_tratado = pasta / "DadosAbertos_LAI_2023_2025_tratado.csv"

# Lista dos arquivos brutos e do formato de cada um (descoberto abrindo os arquivos):
# - encoding: como os caracteres foram gravados. UTF-16 usa 2 bytes por letra; cp850 é o padrão antigo do DOS.
# - aspas_na_linha: True quando a linha inteira vem entre aspas (só acontece no arquivo de 2024).
arquivos = [
    {'ano': 2023, 'nome': '20261107_Pedidos_csv_2023.csv', 'encoding': 'utf-16', 'aspas_na_linha': False},
    {'ano': 2024, 'nome': '20261104_Pedidos_csv_2024.csv', 'encoding': 'cp850',  'aspas_na_linha': True},
    {'ano': 2025, 'nome': '20261106_Pedidos_csv_2025.csv', 'encoding': 'utf-16', 'aspas_na_linha': False},
]


def ler_arquivo(info):
    """Lê um arquivo bruto da LAI de acordo com o seu formato e devolve uma tabela."""
    # quoting=3 manda o Pandas ignorar as aspas (necessário quando a linha inteira está entre aspas).
    # quoting=0 é o comportamento normal. dtype=str lê tudo como texto, para não adivinhar tipos errados.
    tabela = pd.read_csv(pasta / info['nome'], sep=';', encoding=info['encoding'], dtype=str,
                         quoting=3 if info['aspas_na_linha'] else 0, low_memory=False)
    if info['aspas_na_linha']:
        # Remove as aspas que sobraram no primeiro cabeçalho e no primeiro valor de cada linha.
        tabela.columns = [coluna.strip('"') for coluna in tabela.columns]
        tabela['IdPedido'] = tabela['IdPedido'].str.lstrip('"')
    tabela['AnoArquivo'] = info['ano']
    print(f"{info['nome']}: {len(tabela)} linhas")
    return tabela


print("=== Arquivos brutos ===")
# Lê os três arquivos e os empilha em uma tabela só (pd.concat). ignore_index=True renumera as linhas.
df = pd.concat([ler_arquivo(info) for info in arquivos], ignore_index=True)
linhas_brutas = len(df)
print(f"Total: {linhas_brutas} linhas\n")


# ==============================================================================
# 3. TRANSFORMAÇÃO 1: REDUÇÃO DE COLUNAS
# ==============================================================================

# Mantemos as mesmas 15 colunas das versões anteriores, mais a AnoArquivo.
colunas_mantidas = ['IdPedido', 'Esfera', 'OrgaoDestinatario', 'Situacao', 'DataRegistro',
                    'PrazoAtendimento', 'FoiProrrogado', 'FoiReencaminhado', 'FormaResposta',
                    'OrigemSolicitacao', 'IdSolicitante', 'AssuntoPedido', 'DataResposta',
                    'Decisao', 'EspecificacaoDecisao', 'AnoArquivo']
df = df[colunas_mantidas].copy()


# ==============================================================================
# 4. TRANSFORMAÇÃO 2: LIMPEZA DE TEXTO E VALORES AUSENTES
# ==============================================================================

# Remove espaços sobrando no início e no fim de todos os textos (a coluna AnoArquivo é número e fica de fora).
for coluna in df.columns.drop('AnoArquivo'):
    df[coluna] = df[coluna].str.strip()

# Padroniza o traço dos nomes de órgãos: "INSS – Instituto..." (2023/2025) e "INSS ? Instituto..." (2024)
# viram "INSS - Instituto...". Assim o mesmo órgão tem o mesmo nome nos três anos.
df['OrgaoDestinatario'] = (df['OrgaoDestinatario']
                           .str.replace(' – ', ' - ', regex=False)    # – é o código do travessão "–"
                           .str.replace(' ? ', ' - ', regex=False))

# Campos vazios significam "sem informação": preenchemos com um texto claro.
for coluna in ['FormaResposta', 'Decisao', 'EspecificacaoDecisao']:
    df[coluna] = df[coluna].replace('', pd.NA).fillna('Não informado')


# ==============================================================================
# 5. TRANSFORMAÇÃO 3: CONVERSÃO DE DATAS E REMOÇÃO DE REGISTROS INVÁLIDOS
# ==============================================================================

# Converte as datas; textos vazios ou inválidos viram vazio (NaT).
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = pd.to_datetime(df[coluna], format='%d/%m/%Y', errors='coerce')

print("=== Filtros aplicados ===")

# FILTRO A - Esfera Federal (mesmo escopo das versões anteriores).
antes = len(df)
df = df[df['Esfera'] == 'Federal']
print(f"Removidos por não serem esfera Federal: {antes - len(df)}")

# FILTRO B - Sem DataResposta ou sem PrazoAtendimento: sem essas datas não há como saber se atrasou.
antes = len(df)
df = df.dropna(subset=['DataResposta', 'PrazoAtendimento'])
print(f"Removidos por não terem DataResposta ou PrazoAtendimento: {antes - len(df)}")

# FILTRO C - Datas inconsistentes: resposta antes do registro, ou prazo igual/anterior ao registro.
antes = len(df)
df = df[(df['DataResposta'] >= df['DataRegistro']) & (df['PrazoAtendimento'] > df['DataRegistro'])]
print(f"Removidos por datas inconsistentes: {antes - len(df)}")

# FILTRO D - Duplicados: o mesmo IdPedido não pode aparecer duas vezes (nem dentro de um ano, nem entre anos).
antes = len(df)
df = df.drop_duplicates(subset='IdPedido')
print(f"Removidos por IdPedido duplicado: {antes - len(df)}\n")


# ==============================================================================
# 6. CARREGAMENTO: GRAVAÇÃO DO ARQUIVO TRATADO E RESUMO
# ==============================================================================

# Resumo por ano: quantos pedidos sobraram e a taxa de atraso de cada ano (para comparar os anos).
atrasou = df['DataResposta'] > df['PrazoAtendimento']
resumo = pd.DataFrame({'Pedidos': df.groupby('AnoArquivo').size(),
                       'TaxaAtraso': atrasou.groupby(df['AnoArquivo']).mean().round(4)})
print("=== Pedidos e taxa de atraso por ano ===")
print(resumo.to_string(), "\n")

# Volta as datas para o formato dia/mês/ano, igual aos arquivos anteriores.
for coluna in ['DataRegistro', 'PrazoAtendimento', 'DataResposta']:
    df[coluna] = df[coluna].dt.strftime('%d/%m/%Y')

# Salva no mesmo padrão do arquivo de 2024: separador ';', sem número de linha, UTF-8 que o Excel entende.
df.to_csv(arquivo_tratado, sep=';', index=False, encoding='utf-8-sig')

print("=== Resultado ===")
print(f"Linhas brutas: {linhas_brutas} -> Linhas tratadas: {len(df)}")
print(f"Arquivo gravado em: {arquivo_tratado.name}")
print(f"Valores ausentes no total (deve ser 0): {int(df.isna().sum().sum())}")
