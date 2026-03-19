from R0_function import calc_R0
from RC_function import calc_rc_params_nopulse
from data import  opening_data, encontrar_pontos_pulso
from ocv_curve import ocv_curve, identificar_parametros_ocv
from plot import plotar_R0_vs_soc, plot_CCCV, plot_CDCH, plot_MPDCH, plot_RC_curves, plot_ocv_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


## Params
Qn = 3.08
# ---------------------- FILES PATH --------------------------------------------

# Pega o caminho exato da pasta onde o main.py esta
DIRETORIO_ATUAL = Path(__file__).parent

# Volta uma pasta (.parent) e entra na BID003
DIRETORIO_DADOS = DIRETORIO_ATUAL.parent / "BID003"

# Caminhos dos datasets
CCCV = DIRETORIO_DADOS / "BID003_CCCV005.0_02022026.txt"
CDCH = DIRETORIO_DADOS / "BID003_CDch005.0_02022026.txt"
MPDCH = DIRETORIO_DADOS / "BID003_MPDch_24022026.txt"


# ----------------------- GET PARAMS R0 ---------------------------------------


time, voltage, current = opening_data(MPDCH)                # Open MPDCH time, voltage and current data
#R0_mean,R0_median, R0_values_ab, R0_values_cd = calc_R0(MPDCH)        # Open and calc values of R0

# curve, soc_e, tensao_e, soc_linha_suave, tensao_simulada = identificar_parametros_ocv(MPDCH)

# ----------------------- GET PARAMS RC ---------------------------------------

params = calc_rc_params_nopulse(MPDCH)                      # Calculates the values of 2RC params using MPDCH data


#identificar_parametros_ocv(MPDCH)

# ----------------------- PLOTS ---------------------------------------

# plot_CCCV()
# plot_CDCH()
# plot_MPDCH(MPDCH)
# plot_RC_curves()
# plot_ocv_curve(soc_e, tensao_e, soc_linha_suave, tensao_simulada)
# plotar_R0_vs_soc(MPDCH, R0_values_ab, R0_values_cd)

# ----------------------- PRINT VALUES ---------------------------------------

## Print value of R0
# print('\nR0_mean =', R0_mean)
# print('\nR0_median =', R0_median)

df = pd.DataFrame(params)

pd.set_option('display.max_rows', None)

# 2. Agora o print vai mostrar todos os 66 valores de cima a baixo
print(df.mean())


# ----------------------- DEBUG ---------------------------------------

pulses_curr = encontrar_pontos_pulso(current)

# Cria uma lista vazia para armazenar os dados linha por linha
linhas_df = []
'''
for i, pls in enumerate(pulses_curr): 
    # Desempacota os índices
    a_atual, b_atual, c_atual, d_atual, e_atual = pls['a'], pls['b'], pls['c'], pls['d'], pls['e']

    # Adiciona um dicionário com os dados deste pulso à lista
    linhas_df.append({
        'Pulso': i + 1,
        'Índice_b': b_atual,
        'Tensão_b (V)': voltage[b_atual],
        'Corrente_b (A)': current[b_atual]
    })

df_verificacao = pd.DataFrame(linhas_df)

print(df_verificacao)

'''
#         # =========================================================
#         # BLOCO DE VALIDAÇÃO (Imprime os 3 primeiros pulsos)
#         # =========================================================
#         if i < 3:
#             print(f"\n--- VERIFICAÇÃO DO PULSO {i+1} ---")
#             print(f"Ponto 'a' (Repouso) -> Índice: {a_atual:^4} | Tempo: {time[a_atual]:>6.1f} s | Tensão: {voltage[a_atual]:.4f} V | Corrente: {current[a_atual]:.3f} A")
#             print(f"Ponto 'b' (Pulso)   -> Índice: {b_atual:^4} | Tempo: {time[b_atual]:>6.1f} s | Tensão: {voltage[b_atual]:.4f} V | Corrente: {current[b_atual]:.3f} A")
#             print("-" * 60)
#         # =========================================================

#         # ... (O resto do seu código de fatiamento e curve_fit continua aqui) ...