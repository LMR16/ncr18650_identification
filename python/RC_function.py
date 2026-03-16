from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from data import func_1, encontrar_pontos_pulso, opening_data

"""
This program will use scipy curve_fit to aproximate the values of the parâmeters x1, x2, tau1, tau2
of the RC exponential dynamics equation.
Then using theese values it calculates the R1, C1, R2, C2 parameters of the model.

There is one main functions in this file: 'calc_rc_params_nopulse'
This function will receive the MPDCH path as parameter and return a dict with all 2RC values.


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

def calc_rc_params_nopulse(path):

    time, voltage, current = opening_data(path)    
    pulsos = encontrar_pontos_pulso(current) # 66 pulsos
    params = {'R1': [], 'C1': [], 'R2': [], 'C2': []}

    for i, pls in enumerate(pulsos): 
        # Pega o 'a', 'b' e 'c' DO PULSO ATUAL
        a_atual, b_atual, c_atual = pls['a'], pls['b'], pls['c'] 
    
        # Define o fim do relaxamento usando o 'a' DO PRÓXIMO PULSO
        if i < len(pulsos) - 1:
            a_proximo = pulsos[i+1]['a'] 
        else:
            a_proximo = c_atual + 600

        ## fatia o tempo de relaxamento (c -> a_proximo)
        t_fatiado = time[c_atual:a_proximo]
        t_norm = t_fatiado - t_fatiado[0]

        v_curve = voltage[c_atual:a_proximo]
        ye = v_curve[-1]
        y_alvo = ye - v_curve

        chute_inicial = [abs(y_alvo[0])*(0.5), 10.0, abs(y_alvo[0])*(0.5), 100.0]
        limites = (0, np.inf)

        try:
            popt, _ = curve_fit(func_1, t_norm, y_alvo, p0=chute_inicial, bounds=limites, maxfev=10000)
            x1, tau1, x2, tau2 = popt
            
            # FÍSICA DO PULSO ATUAL: Usa a_atual, b_atual e c_atual
            I_pulso = abs(current[b_atual] - current[a_atual]) 
            tempo_pulso = time[c_atual] - time[a_atual]

            if I_pulso > 0:
                R1 = x1 / (I_pulso * (1 - np.exp(-tempo_pulso / tau1)))
                R2 = x2 / (I_pulso * (1 - np.exp(-tempo_pulso / tau2)))
            else:
                R1, R2 = 0, 0

            C1 = tau1 / R1 if R1 > 0 else 0
            C2 = tau2 / R2 if R2 > 0 else 0

            params['R1'].append(R1)
            params['C1'].append(C1)
            params['R2'].append(R2)
            params['C2'].append(C2)
            
        except RuntimeError as e:
            print(f"--- ATENÇÃO: Pulso {i+1} ignorado. Motivo: {e}")

    return params
