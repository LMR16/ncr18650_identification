import numpy as np
from data import encontrar_pontos_pulso, opening_data



def calc_R0(path):
    time, voltage, current = opening_data(path)
    pontos_pulsos = encontrar_pontos_pulso(current)
    
    R0_valores_cd = []
    R0_valores_ab = []

    for i, pts in enumerate(pontos_pulsos):
        a, b, c, d = pts['a'], pts['b'], pts['c'], pts['d']
        
        # --- Cálculo AB (Início do pulso) ---
        delta_v_ab = np.abs(voltage[b] - voltage[a])
        delta_i_ab = np.abs(current[b] - current[a])
        
        # Filtro: só calcula se for um degrau real de corrente (> 1 Ampere)
        if delta_i_ab > 1.0:
            R0_valores_ab.append(delta_v_ab / delta_i_ab)
        else:
            R0_valores_ab.append(np.nan)

        # --- Cálculo CD (Fim do pulso) ---
        delta_v_cd = np.abs(voltage[d] - voltage[c])
        delta_i_cd = np.abs(current[d] - current[c])

        if delta_i_cd > 1.0:
            R0_valores_cd.append(delta_v_cd / delta_i_cd)
        else:
            R0_valores_cd.append(np.nan)
            
    R0_valores_ab = np.array(R0_valores_ab)
    R0_valores_cd = np.array(R0_valores_cd)

    # Calcula as médias ignorando os possíveis NaN (falhas de medição)
    R0_values = np.concatenate((R0_valores_ab, R0_valores_cd))
    R0_mean = np.nanmean(R0_values)
    R0_median = np.nanmedian(R0_values)
    
    return R0_mean, R0_median, R0_valores_ab, R0_valores_cd

