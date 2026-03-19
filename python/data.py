import pandas as pd
import numpy as np

# ---------------------- OPENING DATA ----------------------------------

def opening_data(file_path, as_df=False):
    df = pd.read_csv(file_path, sep=';', usecols=['time', 'voltage', 'current'])
    df = df[df['voltage'] > 2.0].copy()
    df = df.reset_index(drop=True)
    
    if not as_df:
        return df['time'].values, df['voltage'].values, df['current'].values
    
    return df


# ----------------------- FUNCTION 1-------------------------------------

def func_1(t, x1, x2, tau1, tau2):
    return x1 * np.exp(-t / tau1) -x2 * np.exp(-t/tau2)
    

# ----------------------- PULSES --------------------------------------

def encontrar_pontos_pulso(current, threshold=0.1, amostras_e=2):
    """
    Identifica e retorna os índices a, b, c, d, e para todos os pulsos do dataset.
    
    a: instante anterior ao pulso
    b: imediatamente após o pulso
    c: instante anterior ao pulso voltar a 0
    d: imediatamente após o pulso voltar a 0
    e: uma ou duas amostras após d
    """
    ativo = np.abs(current) > threshold 
    mudancas = np.diff(ativo.astype(int))
    
    inicios = np.where(mudancas == 1)[0]
    fins = np.where(mudancas == -1)[0]
    
    pulsos = []
    
    # Pareamento Inteligente
    for ini in inicios:
        # Pega a lista de todos os "fins" que são MAIORES (acontecem depois) que o "ini" atual
        fins_validos = fins[fins > ini]
        
        # Se encontrou algum final pela frente, pega o primeiro deles
        if len(fins_validos) > 0:
            fim = fins_validos[0]
            
            a = ini
            b = ini + 1
            c = fim 
            d = fim + 2 
            e = fim + 10 # Se for usar o 'e' para algo, ele fica um pouco depois do 'd'
            
            pulsos.append({'a': a, 'b': b, 'c': c, 'd': d, 'e': e})
            
    return pulsos