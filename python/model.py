import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from ocv_curve import func_ocv_exp

def simular_bateria_continua(time, voltage, current, params, R0, func_ocv, popt_ocv, Qn):
    """
    Roda a simulação contínua do modelo 2RC e plota a validação.
    Agora aceita uma função OCV de qualquer ordem matemática!
    """
    # 1. Extrai a mediana dos parâmetros
    R1 = np.median(params['R1'])
    R2 = np.median(params['R2'])
    C1 = np.median(params['C1'])
    C2 = np.median(params['C2'])

    # 2. Cria o interpolador de corrente
    interpolador_corrente = interp1d(time, current, bounds_error=False, fill_value=0)

    # 3. A Função EDO
    def edo_bateria(t, X):
        v1, v2, soc = X
        i_t = interpolador_corrente(t)
        
        dv1_dt = (-1 / (R1 * C1)) * v1 + (1 / C1) * i_t
        dv2_dt = (-1 / (R2 * C2)) * v2 + (1 / C2) * i_t
        dsoc_dt = (-1 / (Qn * 3600)) * i_t
        
        return [dv1_dt, dv2_dt, dsoc_dt]

    # 4. Roda o Solver
    print("\nA rodar simulação contínua (solve_ivp)...")
    X0 = [0.0, 0.0, 1.0] # Condições Iniciais
    resultado = solve_ivp(fun=edo_bateria, 
                          t_span=(time[0], time[-1]), 
                          y0=X0, 
                          t_eval=time, 
                          method='RK45')

    v1_sim = resultado.y[0]
    v2_sim = resultado.y[1]
    soc_sim = resultado.y[2]

    # 5. A Equação de Saída (A Mágica da Função Dinâmica)
    soc_seguro = np.clip(soc_sim, 0.001, 0.999) 
    
    # O asterisco (*) desempacota o array popt_ocv inteiro para dentro da função,
    # não importa se ele tem 3, 5 ou 9 valores!
    ocv_sim = func_ocv(soc_seguro, *popt_ocv)
    
    v_terminal_sim = ocv_sim - v1_sim - v2_sim - (R0 * current)

# 6. Cálculo do Erro e Gráficos
    erro_pct = (voltage - v_terminal_sim)/voltage
    rmse_pct = (np.sqrt(np.mean(erro_pct**2)))*100
    
    # Erro absoluto em Volts para calcular os mV
    erro_volts = voltage - v_terminal_sim
    rmse_mv = np.sqrt(np.mean(erro_volts**2)) * 1000.0

    print(f"Validação Concluída! Erro RMSE Final: {rmse_pct:.2f} %")

    plt.figure(figsize=(12, 6))
    plt.plot(time, voltage, 'k-', label='Tensão Real')
    plt.plot(time, v_terminal_sim, 'r--', label='Tensão Simulada')
    plt.title(f"Validação do Modelo 2RC (RMSE: {rmse_mv*1000:.2f} mV)")
    plt.xlabel('Tempo (s)')
    plt.ylabel('Tensão (V)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return v_terminal_sim, soc_sim, erro_volts