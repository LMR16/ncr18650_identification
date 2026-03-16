import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data import opening_data, get_path


CCCV, CDCH, MPDCH = get_path()

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

