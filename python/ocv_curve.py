from data import opening_data, encontrar_pontos_pulso
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d


'''
def func_ocv_exp(soc, k0, k1, alpha1, k2, alpha2):
    """
    OCV(SOC) = K0 + K1*exp(alpha1 * SOC) + K2*exp(alpha2 * SOC)
    """
    return k0 + k1 * np.exp(alpha1 * soc) + k2 * np.exp(alpha2 * soc)
'''

def func_ocv_exp(soc, k1, alpha1, k2, alpha2):
    """
    OCV(SOC) = K1*exp(alpha1 * SOC) + K2*exp(alpha2 * SOC)
    """
    return k1 * np.exp(alpha1 * soc) + k2 * np.exp(alpha2 * soc)

def ocv_curve(time, voltage, current, pulsos):

    # 1. Criar o vetor de SOC (de 1.0 a 0.0)
    samples = len(time)
    soc_global = np.linspace(1, 0, samples)

    soc_pontos_e = []
    tensao_pontos_e = []

    for i, pls in enumerate(pulsos):
        
        # 2. Encontrar o índice do repouso absoluto
        if i < len(pulsos) - 1:
            # O repouso máximo é 1 amostra antes do próximo pulso começar
            indice_repouso = pulsos[i]['a'] - 1 
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

def identificar_parametros_ocv(time, voltage, current, pulsos):
    # Extrair os dados 
    soc_e, tensao_e = ocv_curve(time, voltage, current, pulsos)
    
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
        
        return popt, soc_e, tensao_e, soc_linha_suave, tensao_simulada

    except RuntimeError as e:
        print(f"O algoritmo falhou a convergir: {e}")
        return None
    


# =========================================================================

def identificar_ocv_ordem_n(soc_e, tensao_e, ordem=2):
    """
    Identifica os parâmetros da curva OCV usando uma soma de exponenciais.
    Usa inicializações assimétricas para evitar o colapso do otimizador.
    Retorna os parâmetros, a função OCV e os dados empacotados para plotagem.
    """
    def modelo_ocv(soc, *params):
        y = np.full_like(soc, params[0], dtype=np.float64) 
        for i in range(1, len(params), 2):
            k = params[i]
            alpha = params[i+1]
            y += k * np.exp(alpha * soc)
        return y

    # Chute inicial da tensão base (K0)
    p0 = [3.8]          
    limites_inf = [-50.0] 
    limites_sup = [50.0] 

    chutes_k = [-0.2, 0.4, 0.1, -0.05, 0.02]
    chutes_alpha = [-15.0, -1.5, 2.0, -30.0, 5.0]

    for i in range(ordem):
        # Pega o chute correspondente para não fundir as variáveis
        idx = i % len(chutes_k)
        p0.extend([chutes_k[idx], chutes_alpha[idx]])
        
        # Limites bem abertos para dar liberdade ao otimizador
        limites_inf.extend([-1000.0, -200.0]) 
        limites_sup.extend([1000.0, 200.0])   

    print(f"\n=== Ajuste OCV (Exponencial de Ordem {ordem}) ===")
    
    try:
        # maxfev aumentado drasticamente para dar tempo ao PC de calcular as dobras
        popt, pcov = curve_fit(modelo_ocv, soc_e, tensao_e, 
                               p0=p0, bounds=(limites_inf, limites_sup), 
                               maxfev=500000)
        
        print(f"K0 = {popt[0]:.5f}")
        for i in range(1, len(popt), 2):
            n_termo = (i // 2) + 1
            print(f"K{n_termo} = {popt[i]:.5f} | alpha{n_termo} = {popt[i+1]:.5f}")
            
        erro = (tensao_e - modelo_ocv(soc_e, *popt))/tensao_e
        rmse = (np.sqrt(np.mean(erro**2)))*100
        print(f"RMSE do Ajuste: {rmse:.2f}%")
        print("==================================================")

        soc_linha_suave = np.linspace(0, 1, 200)
        tensao_simulada = modelo_ocv(soc_linha_suave, *popt)

        # --- NOVO: Empacotando dados para enviar ao plots.py ---
        dados_plot = {
            'ordem': ordem,
            'soc_alvo': soc_e,
            'tensao_alvo': tensao_e,
            'soc_suave': soc_linha_suave,
            'linha_simulada': tensao_simulada
        }

        return popt, modelo_ocv, dados_plot

    except RuntimeError:
        print(f"ERRO: O SciPy não conseguiu convergir para a ordem {ordem}.")
        return None, None, None
    

# ======================== OCV CURVE IDENTIFICATION WITH CHARGE AND DISCHARGE CURVES ========================

def processar_ocv_histerese(path_carga, path_descarga, ordem=3):
    """
    Lê os ficheiros de carga e descarga lenta, sincroniza os eixos SOC,
    calcula a OCV termodinâmica (média) e ajusta a equação exponencial.
    
    Retorna:
    - popt: Parâmetros otimizados da equação
    - func_ocv: A função matemática gerada
    - soc_suave, tensao_simulada: Vetores para plotagem
    - Qn_real: A capacidade real da bateria em Ah
    """
    print("\n" + "="*50)
    print("INICIANDO PROCESSAMENTO DA OCV POR HISTERESE")
    print("="*50)

    # 1. Carregar os dados brutos dos ensaios
    time_carg, volt_carg, curr_carg = opening_data(path_carga) 
    time_desc, volt_desc, curr_desc = opening_data(path_descarga) 

    # 2. Descobrir a capacidade real (Qn) integrando a descarga total
    carga_total_As = np.trapezoid(np.abs(curr_desc), time_desc)
    Qn_real = carga_total_As / 3600.0
    print(f"[INFO] Capacidade Real Medida (Qn): {Qn_real:.3f} Ah")

    # 3. Calcular os vetores contínuos de SOC (Coulomb Counting em tempo real)
    # Na descarga, o SOC começa em 1.0 e desce
    q_desc = cumulative_trapezoid(np.abs(curr_desc), time_desc, initial=0)
    soc_descarga = 1.0 - (q_desc / (Qn_real * 3600.0))

    # Na carga, o SOC começa em 0.0 e sobe
    q_carg = cumulative_trapezoid(np.abs(curr_carg), time_carg, initial=0)
    soc_carga = q_carg / (Qn_real * 3600.0)

    # 4. Criar um eixo SOC padronizado para alinhar os dois vetores
    soc_comum = np.linspace(0.01, 0.99, 1000)

    # Ordenar índices para a interpolação funcionar corretamente
    idx_desc = np.argsort(soc_descarga)
    func_desc = interp1d(soc_descarga[idx_desc], volt_desc[idx_desc], 
                         bounds_error=False, fill_value="extrapolate")

    idx_carg = np.argsort(soc_carga)
    func_carg = interp1d(soc_carga[idx_carg], volt_carg[idx_carg], 
                         bounds_error=False, fill_value="extrapolate")

    # 5. Interpolar as tensões e calcular a Média Termodinâmica
    v_desc_sinc = func_desc(soc_comum)
    v_carg_sinc = func_carg(soc_comum)
    ocv_media_real = (v_carg_sinc + v_desc_sinc) / 2.0

    # 6. Chamar a função de Identificação da OCV (que contém o K0)
    popt, modelo_ocv, dados_plot = identificar_ocv_ordem_n(
        soc_comum, ocv_media_real, ordem=ordem
    )

    ordem = dados_plot.get('ordem', 2)
    soc_e = dados_plot['soc_alvo']
    tensao_e = dados_plot['tensao_alvo']
    soc_linha_suave = dados_plot['soc_suave']
    tensao_simulada = dados_plot['linha_simulada']

    # 7. Gráfico Analítico
    if popt is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(soc_comum, v_carg_sinc, 'g--', alpha=0.5, label='Ramo de Carga')
        plt.plot(soc_comum, v_desc_sinc, 'b--', alpha=0.5, label='Ramo de Descarga')
        plt.plot(soc_linha_suave, tensao_simulada, 'r-', linewidth=2, label=f'Modelo Exponencial (Ordem {ordem})')

        plt.title('Identificação da OCV Eliminando a Histerese', fontsize=14)
        plt.xlabel('State of Charge (SOC)', fontsize=12)
        plt.ylabel('Tensão (V)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    return popt, modelo_ocv, soc_linha_suave, tensao_simulada, Qn_real


# ======================== OCV CURVE IDENTIFICATION POLINOMIAL ========================

def identificar_ocv_polinomial(soc_a, tensao_a, soc_b=None, tensao_b=None, ordem=5):
    """
    Gera a curva OCV usando um Polinômio de Ordem 'N'.
    - Modo MPDCH: Passe apenas soc_a e tensao_a.
    - Modo Histerese (C/20): Passe soc_carga, tensao_carga, soc_descarga, tensao_descarga.
    """
    
    # 1. Verifica qual é o modo de operação (MPDCH vs Histerese)
    is_histerese = (soc_b is not None) and (tensao_b is not None)

    if is_histerese:
        print(f"\n=== Identificação OCV Polinomial (Histerese C/20 - Ordem {ordem}) ===")
        # Sincronização e Média Termodinâmica
        soc_comum = np.linspace(0.01, 0.99, 1000)

        # Interpola a Carga (soc_a)
        idx_a = np.argsort(soc_a)
        func_a = interp1d(soc_a[idx_a], tensao_a[idx_a], bounds_error=False, fill_value="extrapolate")

        # Interpola a Descarga (soc_b)
        idx_b = np.argsort(soc_b)
        func_b = interp1d(soc_b[idx_b], tensao_b[idx_b], bounds_error=False, fill_value="extrapolate")

        v_a_sinc = func_a(soc_comum)
        v_b_sinc = func_b(soc_comum)

        soc_alvo = soc_comum
        tensao_alvo = (v_a_sinc + v_b_sinc) / 2.0
        
    else:
        print(f"\n=== Identificação OCV Polinomial (Pontos MPDCH - Ordem {ordem}) ===")
        # Usa os dados diretamente
        soc_alvo = soc_a
        tensao_alvo = tensao_a

    # 2. O Ajuste Polinomial Mágico (Uma única linha de código!)
    # coeficientes = [p_n, p_{n-1}, ..., p_1, p_0]
    coeficientes = np.polyfit(soc_alvo, tensao_alvo, ordem)

    # 3. A Função Empacotada para o seu Simulador
    def modelo_ocv_poli(soc, *coefs):
        # np.polyval avalia o polinômio para qualquer SOC de forma ultra-rápida
        return np.polyval(coefs, soc)

    # 4. Cálculo do Erro e Auditoria
    tensao_simulada = modelo_ocv_poli(soc_alvo, *coeficientes)
    erro = (tensao_alvo - tensao_simulada) / tensao_alvo
    rmse = (np.sqrt(np.mean(erro**2))) * 100
    
    print(f"RMSE do Ajuste: {rmse:.4f} %")
    print(f"Coeficientes do Polinômio: {np.round(coeficientes, 4)}")
    print("==================================================================")

    # 5. Gráficos Adaptativos
    soc_linha_suave = np.linspace(0, 1, 200)
    linha_simulada = modelo_ocv_poli(soc_linha_suave, *coeficientes)

    plt.figure(figsize=(10, 6))
    if is_histerese:
        plt.plot(soc_comum, v_a_sinc, 'g--', alpha=0.4, label='Carga')
        plt.plot(soc_comum, v_b_sinc, 'b--', alpha=0.4, label='Descarga')
        plt.plot(soc_alvo, tensao_alvo, 'k-', linewidth=3, label='OCV Média Real')
    else:
        plt.plot(soc_alvo, tensao_alvo, 'bo', alpha=0.6, label="Dados MPDCH (Pontos de Repouso)")
        
    plt.plot(soc_linha_suave, linha_simulada, 'r-', linewidth=2.5, label=f'Polinômio (Ordem {ordem})')
    plt.title(f'Identificação da OCV (Ajuste Polinomial O({ordem}))', fontsize=14)
    plt.xlabel('State of Charge (SOC)', fontsize=12)
    plt.ylabel('Tensão (V)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return coeficientes, modelo_ocv_poli, soc_linha_suave, linha_simulada


def identificar_ocv_ordem_n_sem_k0(soc_e, tensao_e, ordem=2):
    """
    Identifica os parâmetros da curva OCV usando uma soma de exponenciais.
    Versão SEM K0.
    Inclui máscara robusta do NumPy para limpar múltiplos pontos espúrios.
    """
    
    # =========================================================================
    # FILTRO DE SANIDADE FÍSICA (ROBUSTO)
    # =========================================================================
    # A máscara varre o vetor inteiro e mantém APENAS os pontos reais de repouso.
    limite_seguranca = 3.10  # Tensão mínima realista de repouso (OCV)
    
    mascara_validos = tensao_e >= limite_seguranca
    pontos_removidos = len(tensao_e) - np.sum(mascara_validos)
    
    if pontos_removidos > 0:
        print(f"\n[AVISO] {pontos_removidos} ponto(s) espúrio(s) detetado(s) (V < {limite_seguranca}V). Removendo...")
        soc_e = soc_e[mascara_validos]       
        tensao_e = tensao_e[mascara_validos] 
    # =========================================================================

    def modelo_ocv(soc, *params):
        y = np.zeros_like(soc, dtype=np.float64) 
        for i in range(0, len(params), 2):       
            k = params[i]
            alpha = params[i+1]
            y += k * np.exp(alpha * soc)
        return y

    p0 = []          
    limites_inf = [] 
    limites_sup = [] 

    chutes_k = [-0.2, 0.4, 0.1, -0.05, 0.02]
    chutes_alpha = [-15.0, -1.5, 2.0, -30.0, 5.0]

    for i in range(ordem):
        idx = i % len(chutes_k)
        p0.extend([chutes_k[idx], chutes_alpha[idx]])
        
        limites_inf.extend([-1000.0, -200.0]) 
        limites_sup.extend([1000.0, 200.0])   

    print(f"\n=== Ajuste OCV (Exponencial de Ordem {ordem} - SEM K0) ===")
    
    try:
        popt, pcov = curve_fit(modelo_ocv, soc_e, tensao_e, 
                               p0=p0, bounds=(limites_inf, limites_sup), 
                               maxfev=500000)
        
        for i in range(0, len(popt), 2):
            n_termo = (i // 2) + 1
            print(f"K{n_termo} = {popt[i]:.5f} | alpha{n_termo} = {popt[i+1]:.5f}")
            
        erro = (tensao_e - modelo_ocv(soc_e, *popt))/tensao_e
        rmse = (np.sqrt(np.mean(erro**2)))*100
        print(f"RMSE do Ajuste: {rmse:.2f}%")
        print("==========================================================")

        soc_linha_suave = np.linspace(0, 1, 200)
        tensao_simulada = modelo_ocv(soc_linha_suave, *popt)

        dados_plot = {
            'ordem': ordem,
            'soc_alvo': soc_e,
            'tensao_alvo': tensao_e,
            'soc_suave': soc_linha_suave,
            'linha_simulada': tensao_simulada
        }

        plt.figure(figsize=(10, 6))
        plt.plot(soc_e, tensao_e, 'bo', label="Dados Medidos", alpha=0.6)
        plt.plot(soc_linha_suave, tensao_simulada, 'r-', linewidth=2.5, label=f"Modelo Exponencial O({ordem}) - SEM K0")
        plt.title('Identificação da Tensão de Circuito Aberto (OCV)', fontsize=14)
        plt.xlabel('State of Charge (SOC)', fontsize=12)
        plt.ylabel('Tensão de Repouso (V)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        return popt, modelo_ocv, dados_plot

    except RuntimeError:
        print(f"ERRO: O SciPy não conseguiu convergir para a ordem {ordem}.")
        return None, None, None