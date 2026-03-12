import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def encontrar_pontos_pulso(current, threshold=0.1, amostras_e=2):
    """
    Identifica e retorna os índices a, b, c, d, e para todos os pulsos do dataset.
    
    a: instante anterior ao pulso
    b: imediatamente após o pulso
    c: instante anterior ao pulso voltar a 0
    d: imediatamente após o pulso voltar a 0
    e: uma ou duas amostras após d
    """
    # Array booleano onde True significa que há corrente rodando (pulso)
    is_pulse = np.abs(current) > threshold
    
    # np.diff encontra as transições. 
    # 1 indica que saiu de False (0) para True (1) -> Início do pulso
    # -1 indica que saiu de True (1) para False (0) -> Fim do pulso
    transitions = np.diff(is_pulse.astype(int))

    #np.savetxt('transitions.csv', transitions, delimiter=',')

    starts = np.where(transitions == 1)[0] + 1  ## Point B
    ends = np.where(transitions == -1)[0] + 1   ## Point D
    
    # Tratamento caso o dado comece ou termine no meio de um pulso
    if len(ends) > 0 and len(starts) > 0:
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]
            

    pulsos = []
    for start, end in zip(starts, ends):
        a = start - 1
        b = start
        c = end - 1
        d = end
        e = end + amostras_e
        
        # Validação para não exceder o limite do array
        if a >= 0 and e < len(current):
            pulsos.append({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
            
    return pulsos


def calc_R0(voltage, current):
    """
    Função principal que chama o identificador de pontos e 
    calcula a Resistência Ôhmica (R0) para cada pulso usando os 
    intervalos a-b e.
    """
    pontos_pulsos = encontrar_pontos_pulso(current)
    
    R0_valores_cd = []
    R0_valores_ab = []

    ## loop para o cálculo dos deltas e dos R0s em (a-b) e (c-d)
    for i, pts in enumerate(pontos_pulsos):
        a, b, c, d = pts['a'], pts['b'], pts['c'], pts['d']
        
        # Delta de Tensão e Corrente no degrau inicial do pulso (a -> b)
        delta_v_ab = np.abs(voltage[b] - voltage[a])
        delta_i_ab = np.abs(current[b] - current[a])
        
        if delta_i_ab > 0:
            r0 = delta_v_ab / delta_i_ab
            R0_valores_ab.append(r0)
            #print(f"Pulso {i+1}: Índices (a:{a}, b:{b}, c:{pts['c']}, d:{pts['d']}, e:{pts['e']}) | R0 = {r0:.5f} Ω")

        # Delta de Tensão e Corrente no degrau final do pulso (c -> d)
        delta_v_cd = np.abs(voltage[c] - voltage[d])
        delta_i_cd = np.abs(current[c] - current[d])

        if delta_i_cd > 0:
            r0 = delta_v_cd / delta_i_cd
            R0_valores_cd.append(r0)
            #print(f"Pulso {i+1}: Índices (a:{a}, b:{b}, c:{pts['c']}, d:{pts['d']}, e:{pts['e']}) | R0 = {r0:.5f} Ω")
            
    R0_valores_ab = np.array(R0_valores_ab)
    R0_valores_cd = np.array(R0_valores_cd)

    ## Calculates and prints the mean of R0
    R0_values = np.concatenate((R0_valores_ab, R0_valores_cd))
    R0_mean = np.mean(R0_values)
    
    return R0_mean, R0_valores_ab, R0_valores_cd ## retorna os valores dos R0s já como np.arrays

