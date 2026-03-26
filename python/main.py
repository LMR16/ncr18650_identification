from R0_function import calc_R0
from RC_function import calc_rc_params_nopulse
from data import  opening_data, encontrar_pontos_pulso
from ocv_curve import ocv_curve, processar_ocv_histerese, identificar_ocv_ordem_n, identificar_ocv_polinomial
from plot import plotar_R0_vs_soc, plot_CCCV, plot_CDCH, plot_MPDCH, plot_RC_curves, plot_ocv_curve
from model import simular_bateria_continua
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumulative_trapezoid


# ---------------------- FILES PATH --------------------------------------------

# Pega o caminho exato da pasta onde o main.py esta
DIRETORIO_ATUAL = Path(__file__).parent

# Volta uma pasta (.parent) e entra na BID003
DIRETORIO_DADOS = DIRETORIO_ATUAL.parent / "BID003"

# Caminhos dos datasets
CCCV = DIRETORIO_DADOS / "BID003_CCCV005.0_02022026.txt"
CDCH = DIRETORIO_DADOS / "BID003_CDch005.0_02022026.txt"
MPDCH = DIRETORIO_DADOS / "BID003_MPDch_24022026.txt"


# ----------------------- GET PARAMS ---------------------------------------
R0_mean,R0_median, R0_values_ab, R0_values_cd = calc_R0(MPDCH)   # Open and calc values of R0

## ========================= CALC Qn ================================================= ##

time, voltage, current = opening_data(MPDCH)                     # Open MPDCH time, voltage and current data
time_CDCH, voltage_CDCH, current_CDCH = opening_data(CDCH)
time_CCCV, voltage_CCCV, current_CCCV = opening_data(CCCV)

## Calculo do Qn real usando método dos trapézios para MPDCH
# carga_total_As = np.trapezoid(np.abs(current), time)
# Qn_real = carga_total_As / 3600

## Calculo do Qn real usando método dos trapézios para CDCH
carga_total_As_2 = np.trapezoid(np.abs(current_CDCH), time_CDCH)
Qn_real_2 = carga_total_As_2 / 3600

## Calculo do Qn real usando método dos trapézios para CCCV
carga_total_As_3 = np.trapezoid(np.abs(current_CCCV), time_CCCV)
Qn_real_3 = carga_total_As_3 / 3600

Qn_real = (Qn_real_2 + Qn_real_3)/2


# A) SOC da Descarga: A bateria começou cheia (1.0) e foi esvaziando
q_desc_acumulado = cumulative_trapezoid(np.abs(current_CDCH), time_CDCH, initial=0)
soc_descarga = 1.0 - (q_desc_acumulado / (Qn_real * 3600.0))

# B) SOC da Carga: A bateria começou vazia (0.0) e foi enchendo
q_carg_acumulado = cumulative_trapezoid(np.abs(current_CCCV), time_CCCV, initial=0)
soc_carga = q_carg_acumulado / (Qn_real * 3600.0)

## ========================================================================== ##

pulsos = encontrar_pontos_pulso(current)

params = calc_rc_params_nopulse(MPDCH)                           # Calculates the values of 2RC params using MPDCH data

soc_e, tensao_e = ocv_curve(MPDCH)                               # Get the 'e' points for Soc and voltage
popt_ocv, func_ocv, soc_linha_suave, tensao_simulada = identificar_ocv_ordem_n(soc_e, tensao_e, ordem=2)

popt_2, func_ocv_2, soc_suave_2, tensao_simulada_2, Qn_real_2 = processar_ocv_histerese(CCCV,CDCH, ordem=2)


# ----------------------- PLOTS -----------------------------------------------------

#plot_CCCV()
#plot_CDCH()
#plot_MPDCH(MPDCH)
#plot_RC_curves()
#plot_ocv_curve(soc_e, tensao_e, soc_linha_suave, tensao_simulada)
#plotar_R0_vs_soc(MPDCH, R0_values_ab, R0_values_cd)

# ----------------------- PRINT VALUES ----------------------------------------------


# print('\nR0_mean =', R0_mean)                               #Print value of R0

# print(f"\nCapacidade REAL da bateria: {Qn_real:.3f} Ah")      #Print Q_real of the Battery

# df = pd.DataFrame(params) # Get the calculated params and transforms into a dataframe for better visualization
# print(df.mean())          # Displays the result


# ----------------------- DEBUG ------------------------------------------------------

def auditar_pontos_pulso(time, voltage, current, pulsos, pulso_inicio=1, pulso_fim=2):

    """
    Imprime os valores de Tensão e Corrente no índice exato, um índice antes e um depois,
    para validar o alinhamento perfeito dos pontos a, b, c, d, e.
    Permite escolher um intervalo específico (ex: pulso_inicio=4, pulso_fim=5).
    """
    print("\n" + "="*55)
    print("=== AUDITORIA DE ALINHAMENTO DOS PULSOS ===")
    print("="*55)
    
    # Proteções para garantir que os índices não quebram se passarmos números fora do limite
    total_pulsos = len(pulsos)
    inicio_idx = max(0, pulso_inicio - 1) # Converte de "humano" (1) para índice Python (0)
    fim_idx = min(total_pulsos, pulso_fim)
    
    # Se o intervalo for inválido (ex: início maior que fim)
    if inicio_idx >= fim_idx:
        print("Intervalo inválido. Verifique os números dos pulsos.")
        return

    # Varre apenas o intervalo selecionado
    for i in range(inicio_idx, fim_idx):
        pls = pulsos[i]
        print(f"\n{'='*15} PULSO {i+1} {'='*15}")
        
        # Percorre as chaves na ordem cronológica
        for nome_ponto in ['a', 'b', 'c', 'd', 'e']:
            if nome_ponto not in pls:
                continue
                
            idx = pls[nome_ponto]
            
            print(f"\n[ Ponto '{nome_ponto}' ] -> Índice Base: {idx}")
            print(f"{'Índice':<8} | {'Tempo (s)':<10} | {'Tensão (V)':<12} | {'Corrente (A)'}")
            print("-" * 55)
            
            # Varredura de -1 (antes), 0 (exato) e +1 (depois)
            for offset in [-1, 0, 1]:
                atual = idx + offset
                
                # Proteção para não estourar os limites das listas do ensaio
                if 0 <= atual < len(time):
                    # Coloca uma setinha '->' na linha do índice exato para destacar
                    marca = "-> " if offset == 0 else "   "
                    print(f"{marca}{atual:<5} | {time[atual]:<10.1f} | {voltage[atual]:<10.4f} V | {current[atual]:<10.3f} A")

#auditar_pontos_pulso(time, voltage, current, pulsos, pulso_inicio=1, pulso_fim=2)

# ----------------------- VALIDAÇÃO DO MODELO ---------------------------------------

# v_sim, soc_sim, erro_sim = simular_bateria_continua(
#     time=time, 
#     voltage=voltage, 
#     current=current, 
#     params=params, 
#     R0=R0_median, 
#     func_ocv=func_ocv,
#     popt_ocv=popt_ocv,
#     Qn=Qn_real
# )

# ----------------------- VALIDAÇÃO DO MODELO USANDO PSEUDO OCV ---------------------------------------


## Simulando teste CDCH

# v_sim, soc_sim, erro_sim = simular_bateria_continua(
#     time=time_CDCH, 
#     voltage=voltage_CDCH, 
#     current=current_CDCH, 
#     params=params, 
#     R0=R0_median, 
#     func_ocv=func_ocv_2,
#     popt_ocv=popt_2,
#     Qn=Qn_real_2
# )

## Simulando teste MPDCH

# v_sim, soc_sim, erro_sim = simular_bateria_continua(
#     time=time, 
#     voltage=voltage, 
#     current=current, 
#     params=params, 
#     R0=R0_median, 
#     func_ocv=func_ocv_2,
#     popt_ocv=popt_2,
#     Qn=Qn_real_2
# )

## Aqui ele identifica o OCV com uma funcao polinomial de ordem n a pertir das curvas de MPDCH

popt_poli_mpdch, func_ocv_poli_mpdch, soc_suave, v_suave = identificar_ocv_polinomial(soc_e, tensao_e, ordem=6)

# v_sim, soc_sim, erro_sim = simular_bateria_continua(
#     time=time, 
#     voltage=voltage, 
#     current=current, 
#     params=params, 
#     R0=R0_median, 
#     func_ocv=func_ocv_poli_mpdch,
#     popt_ocv=popt_poli_mpdch,
#     Qn=Qn_real_2
# )


## Aqui ele identifica o OCV com uma funcao polinomial de ordem n a pertir das curvas CCCV e CDCH

popt_poli_hist, func_ocv_poli_hist, soc_suave, v_suave = identificar_ocv_polinomial(
    soc_carga, voltage_CCCV, 
    soc_descarga, voltage_CDCH, 
    ordem=5
)

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=time, 
    voltage=voltage, 
    current=current, 
    params=params, 
    R0=R0_median, 
    func_ocv=func_ocv_poli_hist,
    popt_ocv=popt_poli_hist,
    Qn=Qn_real_2
)