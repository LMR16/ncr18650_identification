import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from R0_function import calc_R0, encontrar_pontos_pulso
from RC_function import get_points_ca, get_points_bc 
from scipy.optimize import curve_fit



## Params

Qn = 3.08

# Pega o caminho exato da pasta onde o main.py esta
DIRETORIO_ATUAL = Path(__file__).parent

# Volta uma pasta (.parent) e entra na BID003
DIRETORIO_DADOS = DIRETORIO_ATUAL.parent / "BID003"

# Caminhos dos datasets
CCCV = DIRETORIO_DADOS / "BID003_CCCV005.0_02022026.txt"
CDCH = DIRETORIO_DADOS / "BID003_CDch005.0_02022026.txt"
MPDCH = DIRETORIO_DADOS / "BID003_MPDch_24022026.txt"

## OPENING DATA

def opening_data(file_path, as_df=False):
    df = pd.read_csv(file_path, sep=';', usecols=['time', 'voltage', 'current'])
    df = df[df['voltage'] > 2.0].copy()
    df = df.reset_index(drop=True)
    
    if not as_df:
        return df['time'].values, df['voltage'].values, df['current'].values
    
    return df


# ----------------------- FUNCTION -------------------------------------

def func_1(t, x1, x2, tau1, tau2):
    return x1 * np.exp(-t / tau1) -x2 * np.exp(-t/tau2)
    

# ----------------------- PLOTING ---------------------------------------


def plot_CCCV():
    ## opening data
    time, voltage, current = opening_data(CCCV)
    samples = len(time)
    soc = np.linspace(0, 1, samples)

    ## Plot OCV
    plt.figure(figsize=(10, 6))

    plt.plot(soc * 100, voltage, label='Charge curve (Pseudo-OCV)', color='blue', linewidth=2)

    plt.title('Voltage vs State of Charge(C/20)', fontsize=14)
    plt.xlabel('State of Charge (SoC) [%]', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 100)
    plt.legend()
    plt.show()

def plot_CDCH():
    ## opening data
    time, voltage, current = opening_data(CDCH)
    samples = len(time)
    soc = np.linspace(1, 0, samples)
    
    ## Plot OCV
    plt.figure(figsize=(10, 6))

    plt.plot(soc * 100, voltage, label='Discharge curve (Pseudo-OCV)', color='Red', linewidth=2)

    plt.title('Voltage vs State of Charge(C/20)', fontsize=14)
    plt.xlabel('State of Charge (SoC) [%]', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(100, 0)
    
    plt.legend()
    plt.show()

def plot_MPDCH():
    ## opening data
    time, voltage, current = opening_data(MPDCH)
    samples = len(time)
    soc = np.linspace(1, 0, samples)
    
    ## Plot OCV
    plt.figure(figsize=(10, 6))

    plt.plot(soc * 100, voltage, label='MPDch curve (Pseudo-OCV)', color='Red', linewidth=2)
    plt.plot(soc * 100, current, label='Current curve (Pseudo-OCV)', color='blue', linewidth=2)

    plt.title('Voltage vs State of Charge(C/20)', fontsize=14)
    plt.xlabel('State of Charge (SoC) [%]', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(100, 0)
    plt.legend()
    plt.show()


def plot_RC_curves(curvas):

    # Prepara a figura
    # Desenha uma curva de cada vez
    for i, curva in enumerate(curvas):
        # Cria um eixo X provisório do tamanho exato desta curva (ex: 0 a 599)
        # Isso faz com que todas as curvas comecem no ponto 0 juntas e fiquem sobrepostas!
        eixo_x_amostras = np.arange(len(curva)) 


    plt.figure(figsize=(10, 6))
    plt.plot(eixo_x_amostras, curva, label=f'Pulso {i+1}')

    plt.title('Curvas de Relaxamento (Sobrepostas)', fontsize=14)
    plt.xlabel('Amostras desde que a corrente zerou', fontsize=12)
    plt.ylabel('Tensão (V)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return None

# ----------------------- RUNNING ---------------------------------------

## Call R0 function
time, voltage, current = opening_data(MPDCH)                        
R0_mean, R0_values_ab, R0_values_cd = calc_R0(voltage, current)      

# Pega a lista com as 66 curvas de tamanhos diferentes
curves_ca = get_points_ca(voltage, current)
pulsos_ca = encontrar_pontos_pulso(current)



# ----------------------- GET PARAMS RC ---------------------------------------

first_pulse = pulsos_ca[0]
c = first_pulse['c']
second_pulse = pulsos_ca[1]
a = second_pulse['a']

## fatia o tempo para o primeiro array e normaliza
t_fatiado = time[c:a]
t_norm = t_fatiado - t_fatiado[0]

#print('t_fatiado', t_fatiado, 't_norm\n', t_norm)

## pega o valor de tensão do primeiro array da curva
v_curve = voltage[c:a] # Corta a tensão EXATAMENTE com o mesmo tamanho do tempo!
# Primeiro set do pulso
ye = v_curve[-1] # ponto 'e' da curva

y_alvo = ye - v_curve

#print('V_curve: ', v_curve, 'Ye: = ', ye, 'y_alvo[0]', y_alvo[0], '')

# 4. Configurar o curve_fit
chute_inicial = [y_alvo[0]*(0.5), 10.0, y_alvo[0]*(0.5), 100.0]
limites = (0, [0.2, np.inf, 0.2, np.inf])


popt, _ = curve_fit(func_1, t_norm, y_alvo, p0=chute_inicial, bounds=limites)
x1, tau1, x2, tau2 = popt

print('X1 = ', x1,'tau 1 = ', tau1,'X2 = ', x2,'Tau 2 = ', tau2)
