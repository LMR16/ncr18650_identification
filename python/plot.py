import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data import opening_data, encontrar_pontos_pulso


# ----------------------- PLOTING ---------------------------------------

def plot_CCCV(path):
    ## opening data
    time, voltage, current = opening_data(path)
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

def plot_CDCH(path):
    ## opening data
    time, voltage, current = opening_data(path)
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

def plot_MPDCH(path):
    ## opening data
    time, voltage, current = opening_data(path)
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


def plotar_R0_vs_soc(path, R0_ab, R0_cd):
    """
    Plota o R0 (média de ab e cd) em relação ao SOC, replicando a Figura 7b do artigo.
    """
    # 1. Carregar os dados originais para obter o tempo e os índices
    time, voltage, current = opening_data(path) ## MPDCH
    pulsos = encontrar_pontos_pulso(current)

    # 2. Calcular o R0 exato de CADA pulso (Equação 41 do artigo)
    # Como são arrays do numpy, a operação é feita elemento a elemento automaticamente
    R0_por_pulso = (R0_ab + R0_cd) / 2

    # 3. Recriar o mapa global de SOC (de 1.0 a 0.0)
    samples = len(time)
    soc_global = np.linspace(1, 0, samples)

    # 4. Extrair o SOC correspondente a cada pulso (no momento 'a', antes da corrente ligar)
    soc_dos_pulsos = []
    for pls in pulsos:
        a_idx = pls['a']
        soc_dos_pulsos.append(soc_global[a_idx])

    # 5. Desenhar o gráfico
    plt.figure(figsize=(8, 5))
    
    # Nota: A Figura 7b do artigo mostra o eixo Y em Ohms (0.06 a 0.14)
    plt.plot(soc_dos_pulsos, R0_por_pulso, color='royalblue',marker='o', linestyle='none', alpha=0.7, markersize=5, label='BID003')
    
    plt.title('Resistência Interna ($R_0$) vs State of Charge (SOC)', fontsize=14)
    plt.xlabel('SOC', fontsize=12)
    plt.ylabel('$R_0$ ($\Omega$)', fontsize=12) # Em Ohms para ficar igual à escala do artigo
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()



def plot_ocv_curve(soc_e, tensao_e, soc_linha_suave, tensao_simulada):

    plt.figure(figsize=(10, 6))
    plt.plot(soc_e, tensao_e, 'bo', label="Dados Medidos (Pontos 'e' MPDch)", alpha=0.6)
    plt.plot(soc_linha_suave, tensao_simulada, 'r-', linewidth=2.5, label="Modelo 2-Exp (Curve Fit)")
    
    plt.title('Identificação da Tensão de Circuito Aberto (OCV)', fontsize=14)
    plt.xlabel('State of Charge (SOC)', fontsize=12)
    plt.ylabel('Tensão de Repouso (V)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return None