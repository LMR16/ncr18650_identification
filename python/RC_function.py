from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from data import encontrar_pontos_pulso, opening_data

"""
This program will use scipy curve_fit to aproximate the values of the parâmeters x1, x2, tau1, tau2
of the RC exponential dynamics equation.
Then using theese values it calculates the R1, C1, R2, C2 parameters of the model.

There is one main functions in this file: 'calc_rc_params_nopulse'
This function will receive the MPDCH path as parameter and return a dict with all 2RC values.


"""

# 1. GARANTIA DA FUNÇÃO: Coloque a func_1 aqui para blindar a matemática
def func_1(t, x1, tau1, x2, tau2):
    return x1 * np.exp(-t / tau1) + x2 * np.exp(-t / tau2)


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

import numpy as np
from scipy.optimize import curve_fit

def calc_rc_params_nopulse(path):
    time, voltage, current = opening_data(path)    
    pulsos = encontrar_pontos_pulso(current)
    params = {'R1': [], 'C1': [], 'R2': [], 'C2': []}

    for i, pls in enumerate(pulsos): 
        a, b, c, d = pls['a'], pls['b'], pls['c'], pls['d']
        a_proximo = pulsos[i+1]['a'] if i < len(pulsos) - 1 else d + 600

        t_fatiado = np.array(time[d:a_proximo], dtype=float)
        if len(t_fatiado) < 10: continue
            
        t_norm = t_fatiado - t_fatiado[0]
        v_curve = np.array(voltage[d:a_proximo], dtype=float)
        y_alvo = v_curve[-1] - v_curve

        # 2. LIMITES LIBERTADOS: Deixamos o tempo (tau) ir até aos 500/5000 segundos!
        p0 = [abs(y_alvo[0])*0.5, 100.0, abs(y_alvo[0])*0.5, 500.0]
        bounds = ([0.0, 1.0, 0.0, 1.0], [1.0, 500.0, 1.0, 5000.0])

        try:
            popt, _ = curve_fit(func_1, t_norm, y_alvo, p0=p0, bounds=bounds)
            x1, tau1, x2, tau2 = popt
            
            if tau1 > tau2:
                tau1, tau2, x1, x2 = tau2, tau1, x2, x1
            
            I_pulso = np.max(np.abs(current[b:c])) 
            tempo_pulso = abs(float(time[c]) - float(time[b]))

            R1 = x1 / (I_pulso * (1 - np.exp(-tempo_pulso / tau1)))
            R2 = x2 / (I_pulso * (1 - np.exp(-tempo_pulso / tau2)))
            C1 = tau1 / R1
            C2 = tau2 / R2

            params['R1'].append(R1)
            params['C1'].append(C1)
            params['R2'].append(R2)
            params['C2'].append(C2)
            
        except Exception:
            pass 

    return params
