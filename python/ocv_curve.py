from data import opening_data, encontrar_pontos_pulso
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



def func_ocv_exp(soc, k0, k1, alpha1, k2, alpha2):
    """
    OCV(SOC) = K0 + K1*exp(alpha1 * SOC) + K2*exp(alpha2 * SOC)
    """
    return k0 + k1 * np.exp(alpha1 * soc) + k2 * np.exp(alpha2 * soc)

def ocv_curve(path):
    time, voltage, current = opening_data(path)
    pulsos = encontrar_pontos_pulso(current)

    # 1. Criar o vetor de SOC (de 1.0 a 0.0)
    samples = len(time)
    soc_global = np.linspace(1, 0, samples)

    soc_pontos_e = []
    tensao_pontos_e = []

    for i, pls in enumerate(pulsos):
        
        # 2. Encontrar o índice do repouso absoluto
        if i < len(pulsos) - 1:
            # O repouso máximo é 1 amostra antes do próximo pulso começar
            indice_repouso = pulsos[i+1]['a'] - 1 
        else:
            # No último pulso, pegamos a última amostra do ensaio inteiro
            indice_repouso = len(voltage) - 1
            
        # 3. Extrair os valores REAIS usando o índice
        soc_pontos_e.append(soc_global[indice_repouso])
        tensao_pontos_e.append(voltage[indice_repouso])

    soc_pontos_e = soc_pontos_e[:-1]
    tensao_pontos_e = tensao_pontos_e[:-1] #exclui o ultimo valor [:-1] pois é intervalo aberto
    
    # Retorna como arrays do numpy para facilitar a matemática depois
    return np.array(soc_pontos_e), np.array(tensao_pontos_e)

def identificar_parametros_ocv(path):
    # Extrair os dados 
    soc_e, tensao_e = ocv_curve(path)
    
    # Configurar o Curve Fit
    chute_inicial = [3.0, 0.5, 2.0, -0.5, -5.0]
    
    # Limites (bounds) para manter a física real:
    # k0 deve ser positivo (entre 2.0V e 4.0V)
    # k1 e k2 livres, alpha1 e alpha2 devem ter limites para não causar "overflow" no exp()
    limites_inferiores = [2.0, -5.0, -20.0, -5.0, -20.0]
    limites_superiores = [4.0,  5.0,  20.0,  5.0,  20.0]
    
    try:
        popt, pcov = curve_fit(func_ocv_exp, soc_e, tensao_e, 
                               p0=chute_inicial, 
                               bounds=(limites_inferiores, limites_superiores),
                               maxfev=50000)
        
        k0, k1, alpha1, k2, alpha2 = popt
        
        print("\n=== Parâmetros OCV Identificados ===")
        print(f"K0 = {k0:.5f}")
        print(f"K1 = {k1:.5f} | alpha1 = {alpha1:.5f}")
        print(f"K2 = {k2:.5f} | alpha2 = {alpha2:.5f}")
        print("====================================")
        
        soc_linha_suave = np.linspace(0, 1, 200)
        tensao_simulada = func_ocv_exp(soc_linha_suave, k0, k1, alpha1, k2, alpha2)
        
        plt.figure(figsize=(10, 6))
        plt.plot(soc_e, tensao_e, 'bo', label="Dados Medidos (Pontos 'e' MPDch)", alpha=0.6)
        plt.plot(soc_linha_suave, tensao_simulada, 'r-', linewidth=2.5, label="Modelo 2-Exp (Curve Fit)")
        
        plt.title('Identificação da Tensão de Circuito Aberto (OCV)', fontsize=14)
        plt.xlabel('State of Charge (SOC)', fontsize=12)
        plt.ylabel('Tensão de Repouso (V)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
        return popt

    except RuntimeError as e:
        print(f"O algoritmo falhou a convergir: {e}")
        return None