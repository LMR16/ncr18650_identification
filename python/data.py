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