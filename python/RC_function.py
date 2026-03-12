from scipy.optimize import curve_fit
from R0_function import encontrar_pontos_pulso
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
This program will use scipy curve_fit to aproximate the values of the parâmeters x1, x2, tau1, tau2
of the RC exponential dynamics equation.

First is the curve when u=0, point d to e

"""

def get_points_ca(voltage, current):
    """
    Extrai as curvas de relaxamento, do ponto "c" 
    até o próximo ponto "a" garantindo que o último pulso
    não apanha o "lixo" do final do dataset.
    """
    pulsos = encontrar_pontos_pulso(current)
    ## Pega os pontos a,b,c,d e e

    todas_as_curvas = []

    for i in range(len(pulsos)):
        c_atual = pulsos[i]['c'] ## usa o primeiro ponto c como referência
        
        # Pega o início do PRÓXIMO pulso (a)
        if i < len(pulsos) - 1:
            a_proximo = pulsos[i+1]['e']
        else:
           tamanho_padrao = len(todas_as_curvas[-1]) if i > 0 else 600
           a_proximo = min(c_atual + tamanho_padrao, len(voltage) - 1)

        # Corta o trecho de relaxamento
        curve = voltage[c_atual:a_proximo] ## Pega do ponto c -> a
        todas_as_curvas.append(curve)
        
        #print(f"Curva do pulso {i+1} tem {len(curve)} pontos.")

    return todas_as_curvas

def get_points_bc(voltage, current):
    """
    Extrai as curvas durante o pulso (b -> c).
    """
    pulsos = encontrar_pontos_pulso(current)
    #print(f"Total de pulsos encontrados: {len(pulsos)}")

    todas_as_curvas = []

    for i in range(len(pulsos)):
        b_atual = pulsos[i]['b']
        c_proximo = pulsos[i]['c']

        v_curve = voltage[b_atual:c_proximo]
        
        todas_as_curvas.append(v_curve)

    return todas_as_curvas


def plot_points_bc(time, voltage, current):
    """
    Extrai as curvas durante o pulso (b -> c) e plota o ensaio.
    """
    pulsos = encontrar_pontos_pulso(current)
    #print(f"Total de pulsos encontrados: {len(pulsos)}")

    samples = len(time)
    soc_global = np.linspace(1, 0, samples) * 100

    todas_as_curvas = []

    # Configuração da figura com 2 eixos Y
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    # 1. PLOTAR O FUNDO (Ensaio Completo)
    ax1.plot(soc_global, voltage, color='lightgray', linewidth=1.5, label='Tensão (Completa)', zorder=1)
    ax2.plot(soc_global, current, color='mistyrose', linewidth=1.5, label='Corrente (Completa)', zorder=1)

    # 2. PLOTAR OS DESTAQUES (Fatias b -> c)
    for i in range(len(pulsos)):
        b_atual = pulsos[i]['b']
        c_proximo = pulsos[i]['c']

        v_curve = voltage[b_atual:c_proximo]
        soc_curve = soc_global[b_atual:c_proximo]
        i_curve = current[b_atual:c_proximo] 
        
        todas_as_curvas.append(v_curve)

        if len(v_curve) > 0:
            # Adiciona a legenda apenas no primeiro pulso para não poluir
            label_v = 'Tensão (Pulso Ativo)' if i == 0 else ""
            label_i = 'Corrente (Pulso Ativo)' if i == 0 else ""

            # Desenha as fatias por cima (zorder=2)
            ax1.plot(soc_curve, v_curve, color='blue', linewidth=2.5, label=label_v, zorder=2)
            ax2.plot(soc_curve, i_curve, color='red', linewidth=2.5, label=label_i, zorder=2)

    # Formatação do Gráfico
    plt.title('Perfil de Descarga com Destaque nas Fases Ativas dos Pulsos (b $\\rightarrow$ c)', fontsize=14)
    
    ax1.set_xlabel('State of Charge (SoC) [%]', fontsize=12)
    ax1.set_ylabel('Tensão (V)', color='blue', fontsize=12)
    ax2.set_ylabel('Corrente (A)', color='red', fontsize=12)
    
    ax1.invert_xaxis() # inverte pois é descarga
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Junta as legendas dos dois eixos numa só e coloca no fundo do gráfico
    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.tight_layout()
    plt.show()

    return None