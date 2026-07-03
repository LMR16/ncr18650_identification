import numpy as np
import pandas as pd
from data import encontrar_pontos_pulso, opening_data
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def calc_R0(voltage, current, pontos_pulsos):
    R0_values = []
    
    for pts in pontos_pulsos:
        a, b, c, d = pts['a'], pts['b'], pts['c'], pts['d']
        
        dv_ab, di_ab = abs(voltage[b] - voltage[a]), abs(current[b] - current[a])
        dv_cd, di_cd = abs(voltage[d] - voltage[c]), abs(current[d] - current[c])
        
        r_ab = (dv_ab / di_ab) if di_ab > 0.5 else np.nan
        r_cd = (dv_cd / di_cd) if di_cd > 0.5 else np.nan
        
        R0_values.append((r_ab + r_cd) / 2)

    R0_values = np.array(R0_values)
    
    return np.nanmean(R0_values), np.nanmedian(R0_values), R0_values


def R0_ident(R0_values):
    # Ignorar possíveis NaNs
    valid_mask = ~np.isnan(R0_values)
    valid_R0 = R0_values[valid_mask]
    
    # x_norm simulando SOC de 1 (início) a 0 (fim)
    x_norm = np.linspace(1, 0, len(valid_R0))

    # NOVO MODELO FÍSICO: R_base + Spike no SOC=0 + Spike no SOC=1
    def model_func(x, R_base, a1, b1, a2, b2):
        # a1 * exp(-b1 * x)        -> Cria a subida na Descarga (x próximo de 0)
        # a2 * exp(-b2 * (1 - x))  -> Cria a subida na Carga (x próximo de 1)
        return R_base + a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * (1.0 - x))
        
    # Chutes Iniciais
    # R_base ~= 0.08 (o fundo do vale)
    # a1 ~= 0.08 (pois R_base + a1 = 0.16 no SOC=0)
    # b1 ~= 15   (taxa de queda rápida para sumir no meio do gráfico)
    # a2 ~= 0.04 (pois R_base + a2 = 0.12 no SOC=1)
    # b2 ~= 15   (taxa de queda rápida para sumir no meio do gráfico)
    p0_guess = [0.08, 0.08, 15.0, 0.04, 15.0]
    
    # Limites
    bounds_inf = [0.05, 0.0, 0.0, 0.0, 0.0]
    bounds_sup = [0.15, 0.5, 100.0, 0.5, 100.0]
    
    try:
        popt, _ = curve_fit(model_func, x_norm, valid_R0, p0=p0_guess, bounds=(bounds_inf, bounds_sup), maxfev=50000)
        
        print(f"\n--- Identificação Exponencial de R0 ---")
        print(f"R_base = {popt[0]:.6f} Ohms")
        print(f"a1 (Pico SOC=0) = {popt[1]:.6f} | b1 = {popt[2]:.2f}")
        print(f"a2 (Pico SOC=1) = {popt[3]:.6f} | b2 = {popt[4]:.2f}")
        print(f"---------------------------------------\n")
        
        # Plotando a curva e os dados
        plt.figure(figsize=(10, 6))
        plt.plot(x_norm, valid_R0, 'bo', label='Dados Medidos', markersize=5, alpha=0.6)
        
        # Gerar uma linha suave para o ajuste
        x_suave = np.linspace(1, 0, 200)
        y_suave = model_func(x_suave, *popt)
        plt.plot(x_suave, y_suave, 'r-', label='Ajuste Exponencial (Físico)', linewidth=2.5)
        
        plt.xlabel('State of Charge - SOC (1 -> 0)', fontsize=12)
        plt.ylabel('Resistência Interna R0 (Ohms)', fontsize=12)
        plt.title('Identificação do R0 em Função do SOC', fontsize=14)
        
        # Inverter o eixo X para que a leitura fique natural (1 na esquerda, 0 na direita)
        #plt.gca().invert_xaxis() 
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        return popt
    except Exception as e:
        print(f"Erro no curve_fit: {e}")
        return None