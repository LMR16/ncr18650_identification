import numpy as np
from data import encontrar_pontos_pulso, opening_data



def calc_R0(path):
    """
    Main function that calculates the R0 ohm resistance for each pulse

    """
    
    time, voltage, current = opening_data(path)
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
    R0_median = np.median(R0_values)
    
    return R0_mean, R0_median, R0_valores_ab, R0_valores_cd ## retorna os valores dos R0s já como np.arrays

