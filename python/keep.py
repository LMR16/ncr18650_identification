
#----------------------- VALIDAÇÃO DO MODELO USANDO PSEUDO OCV ---------------------------------------

popt_ocv_histerese, func_ocv_histerese, soc_suave_histerese, tensao_simulada_histerese, Qn_real_histerese = processar_ocv_histerese(CCCV,CDCH, ordem=2)

popt, modelo_ocv, dados_plot_histerese = identificar_ocv_ordem_n_sem_k0(soc_e, tensao_e, ordem=2)


#Simulando teste CDCH

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=time_CDCH, 
    voltage=voltage_CDCH, 
    current=current_CDCH, 
    params=params, 
    R0=R0_median, 
    func_ocv=func_ocv_histerese,
    popt_ocv=popt_ocv_histerese,
    Qn=Qn_real_histerese
)

# Simulando teste MPDCH

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=time_MPDCH, 
    voltage=voltage_MPDCH, 
    current=current_MPDCH, 
    params=params, 
    R0=R0_median, 
    func_ocv=func_ocv_histerese,
    popt_ocv=popt_ocv_histerese,
    Qn=Q_real_mean
)

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=time_MPDCH, 
    voltage=voltage_MPDCH, 
    current=current_MPDCH, 
    params=params,
    R0=R0_median, 
    func_ocv=modelo_ocv,
    popt_ocv=popt,
    Qn=Q_real_mean
)


#----------------------- VALIDAÇÃO DO MODELO USANDO POLINOMIAL OCV ---------------------------------------


# Aqui ele identifica o OCV com uma funcao POLINOMIAL de ordem n a pertir das curvas de MPDCH

popt_poli_mpdch, func_ocv_poli_mpdch, soc_suave, v_suave = identificar_ocv_polinomial(soc_e, tensao_e, ordem=6)

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=time_MPDCH, 
    voltage=voltage_MPDCH, 
    current=current_MPDCH, 
    params=params, 
    R0=R0_median, 
    func_ocv=func_ocv_poli_mpdch,
    popt_ocv=popt_poli_mpdch,
    Qn=Qn_real_MPDCH
)


# Aqui ele identifica o OCV com uma funcao polinomial de ordem n a pertir das curvas CCCV e CDCH

popt_poli_hist, func_ocv_poli_hist, soc_suave, v_suave = identificar_ocv_polinomial(
    soc_carga, voltage_CCCV, 
    soc_descarga, voltage_CDCH, 
    ordem=6
)

v_sim, soc_sim, erro_sim = simular_bateria_continua(
    time=time_CDCH, 
    voltage=voltage_CDCH, 
    current=current_CDCH, 
    params=params, 
    R0=R0_median, 
    func_ocv=func_ocv_poli_hist,
    popt_ocv=popt_poli_hist,
    Qn=Qn_real_CDCH
)

# ----------------------- DEBUG ------------------------------------------------------

def auditar_pontos_pulso(time, voltage, current, pulsos, pulso_inicio=1, pulso_fim=2):

    """
    Imprime os valores de Tensão e Corrente no índice exato, um índice antes e um depois,
    para validar o alinhamento perfeito dos pontos a, b, c, d, e.
    Permite escolher um intervalo específico (ex: pulso_inicio=4, pulso_fim=5).
    """
    print("\n" + "="*55)
    print("=== AUDITORIA DE ALINHAMENTO DOS PULSOS ===")
    print("="*55)
    
    # Proteções para garantir que os índices não quebram se passarmos números fora do limite
    total_pulsos = len(pulsos)
    inicio_idx = max(0, pulso_inicio - 1) # Converte de "humano" (1) para índice Python (0)
    fim_idx = min(total_pulsos, pulso_fim)
    
    # Se o intervalo for inválido (ex: início maior que fim)
    if inicio_idx >= fim_idx:
        print("Intervalo inválido. Verifique os números dos pulsos.")
        return

    # Varre apenas o intervalo selecionado
    for i in range(inicio_idx, fim_idx):
        pls = pulsos[i]
        print(f"\n{'='*15} PULSO {i+1} {'='*15}")
        
        # Percorre as chaves na ordem cronológica
        for nome_ponto in ['a', 'b', 'c', 'd', 'e']:
            if nome_ponto not in pls:
                continue
                
            idx = pls[nome_ponto]
            
            print(f"\n[ Ponto '{nome_ponto}' ] -> Índice Base: {idx}")
            print(f"{'Índice':<8} | {'Tempo (s)':<10} | {'Tensão (V)':<12} | {'Corrente (A)'}")
            print("-" * 55)
            
            # Varredura de -1 (antes), 0 (exato) e +1 (depois)
            for offset in [-1, 0, 1]:
                atual = idx + offset
                
                # Proteção para não estourar os limites das listas do ensaio
                if 0 <= atual < len(time):
                    # Coloca uma setinha '->' na linha do índice exato para destacar
                    marca = "-> " if offset == 0 else "   "
                    print(f"{marca}{atual:<5} | {time[atual]:<10.1f} | {voltage[atual]:<10.4f} V | {current[atual]:<10.3f} A")

#auditar_pontos_pulso(time, voltage, current, pulsos, pulso_inicio=1, pulso_fim=2)