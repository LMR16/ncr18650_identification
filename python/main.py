from R0_function import calc_R0
from RC_function import calc_rc_params_nopulse
from data import  opening_data
from ocv_curve import ocv_curve, identificar_parametros_ocv
from plot import plotar_R0_vs_soc
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

## Call R0 function
time, voltage, current = opening_data(MPDCH)    
R0_mean, R0_values_ab, R0_values_cd = calc_R0(MPDCH)
plotar_R0_vs_soc(MPDCH, R0_values_ab, R0_values_cd)



# ----------------------- GET PARAMS RC ---------------------------------------


# calc_rc_params_nopulse(time, voltage, current)
params = calc_rc_params_nopulse(MPDCH)

df = pd.DataFrame(params)
#print(df.mean())

identificar_parametros_ocv(MPDCH)
