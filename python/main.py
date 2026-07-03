from pandas import DataFrame
from R0_function import calc_R0, R0_ident
from RC_function import calc_rc_params_nopulse, calc_rc_params_lut
from data import  opening_data, encontrar_pontos_pulso
from ocv_curve import ocv_curve, processar_ocv_histerese, identificar_ocv_ordem_n, identificar_ocv_polinomial, identificar_ocv_ordem_n_sem_k0
from plot import plotar_R0_vs_soc, plot_CCCV, plot_CDCH, plot_MPDCH, plot_RC_curves, plot_ocv_curve
from model import simular_bateria_continua, simular_bateria_continua_lut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumulative_trapezoid

# ---------------------- Classes --------------------------------------------

class BatteryDataset:
    def __init__(self, name, cdch_path, cccv_path, mpdch_path):
        self.name = name
        self.cdch_path = cdch_path
        self.cccv_path = cccv_path
        self.mpdch_path = mpdch_path
        
        # Data arrays
        self.time_cdch = self.voltage_cdch = self.current_cdch = None
        self.time_cccv = self.voltage_cccv = self.current_cccv = None
        self.time_mpdch = self.voltage_mpdch = self.current_mpdch = None
        self.pulsos_mpdch = None

    def load_data(self):
        print(f"[{self.name}] Loading datasets...")
        self.time_cdch, self.voltage_cdch, self.current_cdch = opening_data(self.cdch_path)
        self.time_cccv, self.voltage_cccv, self.current_cccv = opening_data(self.cccv_path)
        self.time_mpdch, self.voltage_mpdch, self.current_mpdch = opening_data(self.mpdch_path)
        
        print(f"[{self.name}] Processing pulse points...")
        self.pulsos_mpdch = encontrar_pontos_pulso(self.current_mpdch)
        print(f"[{self.name}] Data load complete.")

# ---------------------- FILES PATH --------------------------------------------

# Pega o caminho exato da pasta onde o main.py esta
DIRETORIO_ATUAL = Path(__file__).parent

# Volta uma pasta (.parent) e entra na BID003
DIRETORIO_DADOS = DIRETORIO_ATUAL.parent / "BID003"
DIRETORIO_DADOS_4 = DIRETORIO_ATUAL.parent / "BID004"


# Caminhos dos datasets BID003
CCCV = DIRETORIO_DADOS / "BID003_CCCV005.0_02022026.txt"
CCCV_NR = DIRETORIO_DADOS / "BID003_CCCV005.0_27042026.txt"

CDCH = DIRETORIO_DADOS / "BID003_CDch005.0_02022026.txt"
CDCH_NR = DIRETORIO_DADOS / "BID003_CDch005.0_23042026.txt"

MPDCH = DIRETORIO_DADOS / "BID003_MPDch_24022026.txt" ## With relay
MPDCH_NR = DIRETORIO_DADOS / "BID003_MPDch_17042026.txt" ## NR means "No Relay"

# Caminhos dos datasets BID004
CDCH_NR_4 = DIRETORIO_DADOS_4 / "BID004_CDch005.0_07052026.txt"
CCCV_NR_4 = DIRETORIO_DADOS_4 / "BID004_CCCV005.0_08052026.txt"
MPDCH_NR_4 = DIRETORIO_DADOS_4 / "BID004_MPDch_06052026.txt" ## NR means No Relay

# ------------------------------------------------------------------

# Defining test class and loading data
test_data = BatteryDataset("BID004", CDCH_NR_4, CCCV_NR_4, MPDCH_NR_4)
test_data.load_data()


# ----------------------- EXTRACT DATA AND CALCULATE R0 ---------------------------------------

R0_mean, R0_median, R0_values = calc_R0(
    test_data.voltage_mpdch, 
    test_data.current_mpdch, 
    test_data.pulsos_mpdch
)   # Calc values of R0

R0_ident(R0_values)

## ========================= CALC Qn ================================================= ##

'''
The Qn is calculated in two forms

1. With the MPDCh test
2. With the mean of CDCH and CCCV

np.trapezoid integrates the current and gives the Capacity in Ah, we divide by 3600 to get As

'''

# Calculo do Qn usando método dos trapézios para MPDCH
carga_total_As = np.trapezoid(np.abs(test_data.current_mpdch), test_data.time_mpdch)
Qn_MPDCH = carga_total_As / 3600

## Calculo do Qn real usando método dos trapézios para CDCH
#carga_total_As_2 = np.trapezoid(np.abs(test_data.current_cdch), test_data.time_cdch)
carga_total_As_2 = np.trapezoid(np.abs(test_data.current_cdch))

Qn_CDCH = carga_total_As_2 / 3600

## Calculo do Qn real usando método dos trapézios para CCCV
#carga_total_As_3 = np.trapezoid(np.abs(test_data.current_cccv), test_data.time_cccv)
carga_total_As_3 = np.trapezoid(np.abs(test_data.current_cccv))

Qn_CCCV = carga_total_As_3 / 3600

Qn_real_mean = (Qn_CDCH + Qn_CCCV) / 2

print(f"Qn_MPDCH: {Qn_MPDCH:.3f} Ah")
print(f"Qn_CDCH: {Qn_CDCH:.3f} Ah")
print(f"Qn_CCCV: {Qn_CCCV:.3f} Ah")
print(f"Qn_real_mean: {Qn_real_mean:.3f} Ah")


## ========================= CALC SOC ================================================= ##

# A) SOC da Descarga: A bateria começou cheia (1.0) e foi esvaziando
q_desc_acumulado = cumulative_trapezoid(np.abs(test_data.current_cdch), test_data.time_cdch, initial=0)
soc_descarga = 1.0 - (q_desc_acumulado / (Qn_real_mean * 3600.0))

# B) SOC da Carga: A bateria começou vazia (0.0) e foi enchendo
q_carg_acumulado = cumulative_trapezoid(np.abs(test_data.current_cccv), test_data.time_cccv, initial=0)
soc_carga = q_carg_acumulado / (Qn_real_mean * 3600.0)

## ========================= CALC OCV AND RC PARAMS ================================================= ##

'''

'''

params = calc_rc_params_nopulse(test_data.time_mpdch, test_data.voltage_mpdch, test_data.current_mpdch, test_data.pulsos_mpdch)      # Calculates the values of 2RC params using MPDCH data

soc_e, tensao_e = ocv_curve(
    time=test_data.time_mpdch, 
    voltage=test_data.voltage_mpdch, 
    current=test_data.current_mpdch, 
    pulsos=test_data.pulsos_mpdch,
    Qn_Ah = Qn_real_mean
)

df = pd.DataFrame(soc_e, columns=['SOC_Repouso'])

print(df.head())

# ----------------------- PLOTS -----------------------------------------------------

#plot_CCCV()
#plot_CDCH()
#plot_MPDCH(Test)
#plot_RC_curves()
#plot_ocv_curve(soc_e, tensao_e, soc_suave, tensao_simulada)
#plotar_R0_vs_soc(test_data, R0_values, Qn_real_mean)

# ----------------------- PRINT VALUES ----------------------------------------------


# print('\nR0_mean =', R0_mean)       #Print value of R0

# df = pd.DataFrame(params) # Get the calculated params_RC and transforms into a dataframe for better visualization
# print(df.mean())          # Displays the result


# ----------------------- VALIDAÇÃO DO MODELO ---------------------------------------

#popt_ocv_MPDCH, modelo_ocv_MPDCH, dados_plot_MPDCH = identificar_ocv_ordem_n(soc_e, tensao_e, ordem=2)
popt_ocv_MPDCH, modelo_ocv_MPDCH, dados_plot_MPDCH = identificar_ocv_ordem_n_sem_k0(soc_e, tensao_e, ordem=2)


df_params = pd.DataFrame(params)

# 2. Usa a MEDIANA em vez da Média para filtrar os pontos defeituosos
parametros_limpos = df_params.median().to_dict()

print("\n--- Parâmetros RC (Mediana) ---")
for k, v in parametros_limpos.items():
    print(f"{k} = {v:.4f}")

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=test_data.time_mpdch, 
    voltage=test_data.voltage_mpdch, 
    current=test_data.current_mpdch, 
    params=params, # RC params
    R0=R0_median, 
    func_ocv=modelo_ocv_MPDCH,
    popt_ocv=popt_ocv_MPDCH,
    Qn=Qn_real_mean
)

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=test_data.time_cdch, 
    voltage=test_data.voltage_cdch, 
    current=np.abs(test_data.current_cdch),
    params=parametros_limpos, # RC params
    R0=R0_median, 
    func_ocv=modelo_ocv_MPDCH,
    popt_ocv=popt_ocv_MPDCH,
    Qn=Qn_real_mean,
    soc_inicial=1.0
)

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=test_data.time_cccv, 
    voltage=test_data.voltage_cccv, 
    current=-np.abs(test_data.current_cccv),
    params=parametros_limpos, # RC params
    R0=R0_median, 
    func_ocv=modelo_ocv_MPDCH,
    popt_ocv=popt_ocv_MPDCH,
    Qn=Qn_real_mean,
    soc_inicial=0
)


# ----------------------- MODO LUT (LOOK-UP TABLE) ----------------------------------

# print("\n" + "="*50)
# print("INICIANDO SIMULAÇÃO LUT (PARÂMETROS DINÂMICOS)")
# print("="*50)

# params_lut = calc_rc_params_lut(test_data.time_mpdch, test_data.voltage_mpdch, test_data.current_mpdch, test_data.pulsos_mpdch)

# v_sim_lut, soc_sim_lut, erro_sim_lut = simular_bateria_continua_lut(
#     time=test_data.time_mpdch, 
#     voltage=test_data.voltage_mpdch, 
#     current=test_data.current_mpdch, 
#     lut_params=params_lut,
#     func_ocv=modelo_ocv_MPDCH,
#     popt_ocv=popt_ocv_MPDCH,
#     Qn=Qn_real_mean
# )


'''
For this identification we have the following sequence:

1. Choose a dataset and apply it to Test variable (must be MPDCh)
2. It will calculate the R0_mean and the RC params
3. You choose the validation LUT or normal by uncommenting

'''