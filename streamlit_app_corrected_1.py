import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide")
st.title("Painel Avançado de Otimização de Carteira QUANTOVITZ")

# --- Importar funções do financial_analyzer_enhanced_corrected.py ---
try:
    from financial_analyzer_enhanced_corrected import (
        obter_dados_historicos_yf,
        obter_dados_fundamentalistas_detalhados_br,
        calcular_piotroski_f_score_br,
        calcular_value_composite_score,
        get_fama_french_factors,
        calcular_beneish_m_score,
        otimizar_portfolio_scipy,
        calcular_altman_z_score,
        otimizar_portfolio_markowitz_mc,
        calcular_metricas_portfolio, 
        sugerir_alocacao_novo_aporte,
        RISK_FREE_RATE_DEFAULT
    )
    
    st.sidebar.success("Módulo de análise corrigido carregado!")
except ImportError as e:
    st.error(f"Erro ao importar o módulo 'financial_analyzer_enhanced_corrected.py': {e}. Certifique-se de que o arquivo está no diretório correto e todas as dependências (arch, statsmodels) estão instaladas.")
    st.stop()

# --- Entradas do Usuário na Sidebar ---
st.sidebar.header("Parâmetros da Análise")

# 1. Carteira Atual
st.sidebar.subheader("1. Carteira Atual")
ativos_input_str = st.sidebar.text_input("Ativos da carteira (ex: PETR4.SA,VALE3.SA,ITUB4.SA)", "PETR4.SA,VALE3.SA,ITUB4.SA")
pesos_input_str = st.sidebar.text_input("Pesos percentuais da carteira (ex: 40,30,30)", "40,30,30")
valor_total_carteira_atual = st.sidebar.number_input("Valor total da carteira atual (R$)", min_value=0.0, value=100000.0, step=1000.0)

# 2. Novo Aporte (Opcional)
st.sidebar.subheader("2. Novo Aporte (Opcional)")
novo_capital_input = st.sidebar.number_input("Novo capital a ser aportado (R$)", min_value=0.0, value=10000.0, step=100.0)

# 3. Ativos Candidatos (Opcional)
st.sidebar.subheader("3. Ativos Candidatos (Adicionar à Análise)")
candidatos_input_str = st.sidebar.text_input("Ativos candidatos (ex: MGLU3.SA,WEGE3.SA)", "MGLU3.SA,WEGE3.SA")

# 4. Período de Análise de Dados Históricos
st.sidebar.subheader("4. Período de Análise Histórica")
start_date_analise_input = st.sidebar.date_input("Data Inicial para Dados Históricos", datetime.today() - timedelta(days=5*365))
end_date_analise_input = st.sidebar.date_input("Data Final para Dados Históricos", datetime.today())
start_date_analise = start_date_analise_input.strftime("%Y-%m-%d")
end_date_analise = end_date_analise_input.strftime("%Y-%m-%d")

# 5. Taxa Livre de Risco
st.sidebar.subheader("5. Taxa Livre de Risco (Anual)")
taxa_livre_risco_input = st.sidebar.number_input("Taxa Livre de Risco (ex: 0.02 para 2%)", min_value=0.0, max_value=1.0, value=RISK_FREE_RATE_DEFAULT, step=0.001, format="%.3f")

# 6. Configurações de Otimização Avançada
st.sidebar.subheader("6. Configurações de Otimização Avançada")
vc_metrics_selection = st.sidebar.multiselect(
    "Métricas para Value Composite Score (VC2/VC6)",
    options=[
        'trailingPE', 'priceToBook', 'enterpriseToEbitda', 
        'dividendYield', 'returnOnEquity', 'netMargin', 
        'forwardPE', 'marketCap'
    ],
    default=[
        'trailingPE', 'priceToBook', 'enterpriseToEbitda', 
        'dividendYield', 'returnOnEquity', 'netMargin'
    ]
)
VC_METRIC_DIRECTIONS = {
    'trailingPE': 'lower_is_better', 
    'priceToBook': 'lower_is_better', 
    'enterpriseToEbitda': 'lower_is_better',
    'dividendYield': 'higher_is_better', 
    'returnOnEquity': 'higher_is_better',
    'netMargin': 'higher_is_better',
    'forwardPE': 'lower_is_better',
    'marketCap': 'lower_is_better'
}
vc_metrics_config = {metric: VC_METRIC_DIRECTIONS[metric] for metric in vc_metrics_selection if metric in VC_METRIC_DIRECTIONS}

min_piotroski_score = st.sidebar.slider(
    "Piotroski F-Score Mínimo para Inclusão de Ativos",
    0, 9, 0,
    help="Ativos com score abaixo deste valor podem ser excluídos da otimização avançada. 0 para não filtrar."
)

mostrar_detalhes_piotroski = st.sidebar.checkbox("Mostrar detalhes dos critérios do Piotroski F-Score")
            
# 7. Restrições de Peso na Otimização
st.sidebar.subheader("7. Restrições de Alocação (Otimização)")
min_aloc_global = st.sidebar.slider(
    "Alocação Mínima Global por Ativo (%)", 0, 40, 0, 
    help="Restrição inferior para cada ativo na carteira otimizada."
) / 100.0
max_aloc_global = st.sidebar.slider(
    "Alocação Máxima Global por Ativo (%)", 10, 100, 100, 
    help="Restrição superior para cada ativo na carteira otimizada."
) / 100.0

manter_pesos_atuais_opcao = st.sidebar.selectbox(
    "Considerar Pesos Atuais na Otimização Avançada",
    options=["Não considerar", "Como ponto de partida", "Como restrição inferior aproximada", "Como restrição de intervalo"],
    index=1, help="Define como os pesos da carteira atual informada são usados na otimização avançada."
)
tolerancia_peso_atual = st.sidebar.slider(
    "Tolerância para Restrição de Peso Atual (% do peso atual)",
    0, 100, 20,
    help="Usado se 'Como restrição de intervalo' for selecionado. Ex: 20% -> peso_atual*0.8 a peso_atual*1.2."
)

# --- Filtro Quant Value para novo aporte (integração com Markowitz) ---
st.sidebar.subheader("Filtro Quant Value para Novo Aporte")
top_n_quant_value = st.sidebar.number_input(
    "Comprar apenas os Top N ativos em Quant Value Score (0 = todos)", 
    min_value=0, max_value=20, value=0
)
min_quant_value = st.sidebar.slider(
    "Nota mínima Quant Value Score para compra (0 = sem filtro)", 
    min_value=0.0, max_value=1.0, value=0.0, step=0.01
)

run_analysis = st.sidebar.button("Executar Análise Avançada")

def key_to_str(k):
    if isinstance(k, tuple):
        return k[0]
    return str(k)
    
def plot_efficient_frontier_comparative(fronteiras_data, portfolios_otimizados, carteira_atual_metricas=None):
    if not any(f["pontos"] for f in fronteiras_data):
        st.write("Não foi possível gerar dados para a Fronteira Eficiente.")
        return

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    idx_color = 0

    for i, data in enumerate(fronteiras_data):
        nome_fronteira = data["nome"]
        pontos = data["pontos"]
        if pontos:
            df_fronteira = pd.DataFrame(pontos)
            fig.add_trace(go.Scatter(x=df_fronteira['volatilidade']*100, y=df_fronteira['retorno']*100,
                                     mode='markers', name=f'Simulações ({nome_fronteira})',
                                     marker=dict(color=df_fronteira['sharpe'], colorscale='Viridis', showscale=(i==0), size=5, line=dict(width=0),
                                                 colorbar=dict(title='Sharpe Ratio')),
                                     text=[f"Sharpe: {s:.2f}<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in p.items()})[:100]}..." 
                                           for s, p in zip(df_fronteira['sharpe'], df_fronteira['pesos'])]))
    if carteira_atual_metricas:
        fig.add_trace(go.Scatter(x=[carteira_atual_metricas['volatilidade']*100], y=[carteira_atual_metricas['retorno_esperado']*100],
                                 mode='markers+text', name='Carteira Atual',
                                 marker=dict(color='black', size=12, symbol='diamond-tall'),
                                 text="Atual", textposition="top center",
                                 hovertext=f"Carteira Atual<br>Sharpe: {carteira_atual_metricas['sharpe_ratio']:.2f}"))

    portfolio_symbols = ['star', 'circle', 'cross', 'triangle-up', 'pentagon']
    for i, portfolio_info in enumerate(portfolios_otimizados):
        nome = portfolio_info['nome']
        portfolio = portfolio_info['data']
        if portfolio:
            fig.add_trace(go.Scatter(x=[portfolio['volatilidade']*100], y=[portfolio['retorno_esperado']*100],
                                     mode='markers+text', name=nome,
                                     marker=dict(color=colors[idx_color % len(colors)], size=14, symbol=portfolio_symbols[i % len(portfolio_symbols)]),
                                     text=nome.split(" ")[0],
                                     textposition="bottom right",
                                     hovertext=f"{nome}<br>Sharpe: {portfolio.get('sharpe_ratio', 0):.2f}<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in portfolio['Pesos'].items()})}"))
            idx_color +=1

    fig.update_layout(title='Fronteiras Eficientes e Carteiras Otimizadas',
                      xaxis_title='Volatilidade Anualizada (%)',
                      yaxis_title='Retorno Esperado Anualizado (%)',
                      legend_title_text='Legenda',
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_pie_chart(weights_dict, title):
    if not weights_dict or sum(weights_dict.values()) == 0:
        return
    df_pie = pd.DataFrame(list(weights_dict.items()), columns=['Ativo', 'Peso'])
    df_pie = df_pie[df_pie['Peso'] > 1e-4] # Filter out very small weights for cleaner pie chart
    fig = px.pie(df_pie, values='Peso', names='Ativo', title=title, hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_comparative_table(carteiras_data):
    if not carteiras_data:
        st.write("Não há dados de carteiras para comparar.")
        return
    df_comparativo = pd.DataFrame(carteiras_data)
    df_comparativo_display = df_comparativo[['Nome', 'Retorno Esperado (%)', 'Volatilidade (%)', 'Sharpe Ratio']].copy()
    df_comparativo_display = df_comparativo_display.set_index('Nome')
    st.subheader("Tabela Comparativa de Desempenho")
    st.dataframe(df_comparativo_display.style.format("{:.2f}"))
    st.subheader("Composição Detalhada das Carteiras (%)")
    todos_ativos_pesos = set()
    for c_data in carteiras_data:
        c = c_data.get("Dados")
        if c and 'Pesos' in c and isinstance(c['Pesos'], dict):
            todos_ativos_pesos.update(c['Pesos'].keys())
    pesos_data_list = []
    for c_data in carteiras_data:
        c = c_data.get("Dados")
        row = {'Nome': c_data['Nome']}
        if c and 'Pesos' in c and isinstance(c['Pesos'], dict):
            for ativo in todos_ativos_pesos:
                row[ativo] = c['Pesos'].get(ativo, 0) * 100
        else:
            for ativo in todos_ativos_pesos:
                row[ativo] = 0
        pesos_data_list.append(row)
    df_pesos_detalhados = pd.DataFrame(pesos_data_list).set_index('Nome')
    st.dataframe(df_pesos_detalhados.style.format("{:.2f}"))
    
if run_analysis:
    st.header("Resultados da Análise Avançada")
    ativos_carteira_lista_raw = [s.strip().upper() for s in ativos_input_str.split(',') if s.strip()]
    try:
        pesos_carteira_lista_pct_raw = [float(p.strip()) for p in pesos_input_str.split(',') if p.strip()]
        if len(ativos_carteira_lista_raw) != len(pesos_carteira_lista_pct_raw):
            st.error("O número de ativos e pesos na carteira atual deve ser o mesmo.")
            st.stop()
        if not np.isclose(sum(pesos_carteira_lista_pct_raw), 100.0, atol=0.1):
            st.warning(f"A soma dos pesos da carteira atual ({sum(pesos_carteira_lista_pct_raw):.2f}%) não é 100%. Ajuste ou os resultados podem ser inconsistentes.")
        ativos_carteira_lista = []
        pesos_carteira_lista_pct = []
        for ativo, peso_pct in zip(ativos_carteira_lista_raw, pesos_carteira_lista_pct_raw):
            if peso_pct > 1e-4:
                ativos_carteira_lista.append(ativo)
                pesos_carteira_lista_pct.append(peso_pct)
        pesos_carteira_decimal = {ativo: peso/100.0 for ativo, peso in zip(ativos_carteira_lista, pesos_carteira_lista_pct)}
        carteira_atual_composicao_valores = {ativo: pesos_carteira_decimal[ativo] * valor_total_carteira_atual for ativo in ativos_carteira_lista}
    except ValueError:
        st.error("Os pesos da carteira atual devem ser números.")
        st.stop()
    ativos_candidatos_lista = [s.strip().upper() for s in candidatos_input_str.split(',') if s.strip()]
    todos_ativos_analise = sorted(list(set(ativos_carteira_lista + ativos_candidatos_lista)))
    if not todos_ativos_analise:
        st.error("Nenhum ativo fornecido para análise (carteira atual ou candidatos).")
        st.stop()
    st.info(f"Ativos para análise: {', '.join(todos_ativos_analise)}\nPeríodo histórico: {start_date_analise} a {end_date_analise}")
    df_retornos_historicos = pd.DataFrame()
    df_fundamental_completo = pd.DataFrame()
    fama_french_factors_df = pd.DataFrame()
    with st.spinner("Coletando e processando dados... Por favor, aguarde."):
        df_retornos_historicos = obter_dados_historicos_yf(todos_ativos_analise, start_date_analise, end_date_analise)
        if df_retornos_historicos.empty or df_retornos_historicos.shape[0] < 60:
            st.error(f"Não foi possível obter dados históricos suficientes para {', '.join(todos_ativos_analise)} no período especificado. Verifique os tickers e o período.")
            st.stop()
        df_fundamental_completo = obter_dados_fundamentalistas_detalhados_br(todos_ativos_analise)
        if not df_fundamental_completo.empty:
            df_fundamental_completo.set_index('ticker', inplace=True, drop=False)
# Garantindo que a função sempre retorna (score, detalhes)
            df_fundamental_completo["Piotroski_F_Score"], df_fundamental_completo["Piotroski_F_Detalhes"] = zip(
                *df_fundamental_completo.apply(
                    lambda row: calcular_piotroski_f_score_br(row, verbose=True)[:2],
                    axis=1
                )
            )
            df_fundamental_completo['Quant_Value_Score'] = calcular_value_composite_score(df_fundamental_completo, vc_metrics_config)
            df_fundamental_completo['Altman_Z_Score'] = df_fundamental_completo.apply(calcular_altman_z_score, axis=1)
            df_fundamental_completo['Beneish_M_Score'] = df_fundamental_completo.apply(calcular_beneish_m_score, axis=1)
            st.subheader("Dados Fundamentalistas e Scores")
            colunas_desejadas = [
                'ticker', 'Piotroski_F_Score', 'Quant_Value_Score',
                'Altman_Z_Score', 'Beneish_M_Score', 'enterpriseToEbitda', 'netMargin'
            ]
            colunas_presentes = [c for c in colunas_desejadas if c in df_fundamental_completo.columns]
            st.dataframe(df_fundamental_completo[colunas_presentes])
            if mostrar_detalhes_piotroski:
                detalhes_df = df_fundamental_completo['Piotroski_F_Detalhes'].apply(pd.Series)
                detalhes_df['ticker'] = df_fundamental_completo['ticker'].values
                st.dataframe(detalhes_df.set_index('ticker'))
        else:
            st.warning("Não foi possível obter dados fundamentalistas. A otimização avançada pode ser limitada.")

        ff_start_date = (pd.to_datetime(start_date_analise) - timedelta(days=30)).strftime("%Y-%m-%d")
        fama_french_factors_df = get_fama_french_factors(ff_start_date, end_date_analise)
        if fama_french_factors_df.empty:
            st.warning("Não foi possível obter dados para os fatores Fama-French. Estimativas de Alpha/Beta não serão realizadas.")
    carteiras_comparativo_lista = []
    fronteiras_plot_data = []
    portfolios_otimizados_plot_data = []
    carteira_atual_metricas_plot = None
    if ativos_carteira_lista and not df_retornos_historicos.empty:
        retornos_carteira_atual = df_retornos_historicos[ativos_carteira_lista]
        if retornos_carteira_atual.shape[1] == len(ativos_carteira_lista) and not retornos_carteira_atual.isnull().values.any():
            pesos_np = np.array([pesos_carteira_decimal[ativo] for ativo in ativos_carteira_lista])
            ret_med_atuais = retornos_carteira_atual.mean() * 252
            mat_cov_atuais = retornos_carteira_atual.cov() * 252
            if not ret_med_atuais.empty and not mat_cov_atuais.empty:
                ret_atual, vol_atual, sharpe_atual = calcular_metricas_portfolio(pesos_np, ret_med_atuais, mat_cov_atuais, taxa_livre_risco_input)
                carteira_atual_metricas_plot = {
                    'retorno_esperado': ret_atual, 
                    'volatilidade': vol_atual, 
                    'sharpe_ratio': sharpe_atual
                }
                carteiras_comparativo_lista.append({
                    'Nome': 'Carteira Atual',
                    'Retorno Esperado (%)': ret_atual * 100,
                    'Volatilidade (%)': vol_atual * 100,
                    'Sharpe Ratio': sharpe_atual,
                    'Dados': { 'Pesos': pesos_carteira_decimal }
                })
            else:
                st.warning("Não foi possível calcular métricas para a carteira atual devido a dados insuficientes após o processamento.")
        else:
            st.warning(
                f"Dados de retorno ausentes para alguns ativos da carteira atual: {set(ativos_carteira_lista) - set(df_retornos_historicos.columns)}. "
                "Métricas da carteira atual não calculadas."
            )
    ativos_para_otimizar = todos_ativos_analise.copy()
    if not df_fundamental_completo.empty and 'Piotroski_F_Score' in df_fundamental_completo.columns and min_piotroski_score > 0:
        ativos_filtrados_piotroski = df_fundamental_completo[df_fundamental_completo['Piotroski_F_Score'] >= min_piotroski_score].index.tolist()
        if ativos_filtrados_piotroski:
            st.info(f"Ativos após filtro Piotroski (>= {min_piotroski_score}): {', '.join(ativos_filtrados_piotroski)}")
            ativos_para_otimizar = [
                a for a in ativos_filtrados_piotroski
                if a in df_retornos_historicos.columns and not df_retornos_historicos[a].isnull().all().item()
            ]
            if not ativos_para_otimizar:
                st.warning("Nenhum ativo restou após o filtro Piotroski e verificação de dados de retorno. Usando todos os ativos para otimização.")
                ativos_para_otimizar = [
                    a for a in todos_ativos_analise
                    if a in df_retornos_historicos.columns and not df_retornos_historicos[a].isnull().all().item()
                ]
        else:
            st.warning(f"Nenhum ativo atendeu ao critério Piotroski F-Score >= {min_piotroski_score}. Usando todos os ativos para otimização.")
            ativos_para_otimizar = [
                a for a in todos_ativos_analise
                if a in df_retornos_historicos.columns and not df_retornos_historicos[a].isnull().all().item()
            ]
    else:
         ativos_para_otimizar = [
             a for a in todos_ativos_analise
             if a in df_retornos_historicos.columns and not df_retornos_historicos[a].isnull().all().item()
         ]
    if not ativos_para_otimizar:
        st.error("Nenhum ativo válido restante para otimização após filtros e verificação de dados. Análise interrompida.")
        st.stop()
    st.write(f"Ativos efetivamente usados na otimização: {', '.join(ativos_para_otimizar)}")
    df_retornos_otim = df_retornos_historicos[ativos_para_otimizar].copy().dropna(how='any')
    if df_retornos_otim.shape[0] < 30:
        st.error("Dados de retorno insuficientes para os ativos de otimização após limpeza. Análise interrompida.")
        st.stop()
    # Monte Carlo Markowitz robusto
    ativos_validos_mc = [
        a for a in ativos_para_otimizar
        if a in df_retornos_otim.columns and not df_retornos_otim[a].isnull().all().item()
    ]
    if not ativos_validos_mc:
        st.error("Nenhum ativo válido com dados históricos suficientes para Monte Carlo.")
        st.stop()
    df_retornos_mc = df_retornos_otim[ativos_validos_mc].copy()
    with st.spinner("Executando Otimização Markowitz (Monte Carlo)..."):
        portfolio_markowitz_mc, fronteira_mc_pontos = otimizar_portfolio_markowitz_mc(
            ativos=ativos_validos_mc,
            df_retornos_historicos=df_retornos_mc,
            taxa_livre_risco=taxa_livre_risco_input,
            num_portfolios_simulados=100000
        )
        if portfolio_markowitz_mc:
            carteiras_comparativo_lista.append({
                'Nome': 'Otimizada Markowitz (Sharpe MC)',
                'Retorno Esperado (%)': portfolio_markowitz_mc['retorno_esperado'] * 100,
                'Volatilidade (%)': portfolio_markowitz_mc['volatilidade'] * 100,
                'Sharpe Ratio': portfolio_markowitz_mc['sharpe_ratio'],
                'Dados': { 'Pesos': portfolio_markowitz_mc['pesos'] }
            })
            portfolios_otimizados_plot_data.append({'nome': 'Markowitz MC Sharpe', 'data': {**portfolio_markowitz_mc, 'Pesos': portfolio_markowitz_mc['pesos']}})
            if fronteira_mc_pontos:
                fronteiras_plot_data.append({'nome': 'Markowitz MC', 'pontos': fronteira_mc_pontos})
        else:
            st.warning("Não foi possível otimizar a carteira com Markowitz (Monte Carlo).")
    with st.spinner("Executando Otimização Avançada (Quant Value & Econometria)..."):
        restricoes_pesos_otim = {}
        for ativo_o in ativos_para_otimizar:
            restricoes_pesos_otim[ativo_o] = (min_aloc_global, max_aloc_global)
        pesos_iniciais_otim = None
        if manter_pesos_atuais_opcao != "Não considerar" and pesos_carteira_decimal:
            pesos_atuais_para_otim = {a: pesos_carteira_decimal.get(a, 0.0) for a in ativos_para_otimizar}
            soma_pesos_atuais_otim = sum(pesos_atuais_para_otim.values())
            if np.isclose(soma_pesos_atuais_otim, 1.0) and soma_pesos_atuais_otim > 0:
                pesos_iniciais_otim = pesos_atuais_para_otim
            elif soma_pesos_atuais_otim > 0:
                pesos_iniciais_otim = {a: p/soma_pesos_atuais_otim for a,p in pesos_atuais_para_otim.items()}
                st.info(f"Pesos atuais para {manter_pesos_atuais_opcao} foram normalizados para os ativos em otimização.")
            else:
                st.info(f"Não foi possível usar pesos atuais como ponto de partida (soma zero ou vazios para ativos em otimização). Usando pesos iguais.")
            if manter_pesos_atuais_opcao == "Como restrição inferior aproximada":
                for ativo_o, peso_atual_o in pesos_atuais_para_otim.items():
                    restricoes_pesos_otim[ativo_o] = (max(min_aloc_global, peso_atual_o * (1-tolerancia_peso_atual*2) ), restricoes_pesos_otim[ativo_o][1])
            elif manter_pesos_atuais_opcao == "Como restrição de intervalo":
                for ativo_o, peso_atual_o in pesos_atuais_para_otim.items():
                    min_restrito = max(0.0, peso_atual_o * (1 - tolerancia_peso_atual/100.0))
                    max_restrito = min(1.0, peso_atual_o * (1 + tolerancia_peso_atual/100.0))
                    restricoes_pesos_otim[ativo_o] = (max(min_aloc_global, min_restrito), min(max_aloc_global, max_restrito))
        df_fund_otim = None
        if not df_fundamental_completo.empty:
            df_fund_otim = df_fundamental_completo.reindex(ativos_para_otimizar).dropna(subset=['ticker'])
            if df_fund_otim.empty:
                st.warning("Dados fundamentalistas não encontrados para os ativos selecionados para otimização. Otimização avançada usará apenas dados históricos.")
                df_fund_otim = None
        portfolio_avancado, _ = otimizar_portfolio_scipy(
            ativos=ativos_para_otimizar,
            df_retornos_historicos=df_retornos_otim,
            df_fundamental_completo=df_fund_otim, 
            fama_french_factors=fama_french_factors_df if not fama_french_factors_df.empty else None,
            taxa_livre_risco=taxa_livre_risco_input,
            pesos_atuais=pesos_iniciais_otim if manter_pesos_atuais_opcao == "Como ponto de partida" else None,
            restricoes_pesos_min_max=restricoes_pesos_otim,
            objetivo='max_sharpe'
        )
        if portfolio_avancado:
            carteiras_comparativo_lista.append({
                'Nome': 'Otimizada Avançada (Quant+Econo)',
                'Retorno Esperado (%)': portfolio_avancado['retorno_esperado'] * 100,
                'Volatilidade (%)': portfolio_avancado['volatilidade'] * 100,
                'Sharpe Ratio': portfolio_avancado['sharpe_ratio'],
                'Dados': { 'Pesos': portfolio_avancado['pesos'] }
            })
            portfolios_otimizados_plot_data.append({'nome': 'Avançada Quant+Econo', 'data': {**portfolio_avancado, 'Pesos': portfolio_avancado['pesos']}})
            st.subheader("Detalhes da Otimização Avançada")
            if portfolio_avancado.get('garch_volatilities'):
                st.write("Volatilidades Anualizadas GARCH (estimadas):")
                st.json({k: f"{v*100:.2f}%" for k,v in portfolio_avancado['garch_volatilities'].items() if pd.notna(v)})
            if portfolio_avancado.get('alphas'):
                st.write("Alphas Anualizados (vs Fama-French 4 Fatores Proxy):")
                st.json({k: f"{v*100:.2f}%" for k,v in portfolio_avancado['alphas'].items() if pd.notna(v)})
            if portfolio_avancado.get('arima_forecasts'):
                st.write("Previsões de Retorno Diário ARIMA (próximo período):")
                st.json({k: f"{v*100:.4f}%" for k,v in portfolio_avancado['arima_forecasts'].items() if pd.notna(v)})
        else:
            st.warning("Não foi possível otimizar a carteira com o modelo avançado.")
    st.header("Comparativo de Carteiras")
    if carteiras_comparativo_lista:
        display_comparative_table(carteiras_comparativo_lista)
        plot_efficient_frontier_comparative(fronteiras_plot_data, portfolios_otimizados_plot_data, carteira_atual_metricas_plot)
        cols_pie = st.columns(len(carteiras_comparativo_lista))
        for i, c_data in enumerate(carteiras_comparativo_lista):
            c = c_data.get("Dados")
            if c and 'Pesos' in c:
                with cols_pie[i]:
                    plot_portfolio_pie_chart(c['Pesos'], c_data['Nome'])
    else:
        st.info("Nenhuma carteira para comparar.")

    # ---- SUGESTÃO DE ALOCAÇÃO PARA NOVO APORTE: Fronteira Markowitz + Quant Value ----
    # ... (processamento da análise, coleta de dados, otimizações, etc.) ...
    # SUGESTÃO DE ALOCAÇÃO PARA NOVO APORTE: Fronteira Markowitz + Quant Value
    if novo_capital_input > 0 and carteiras_comparativo_lista and not df_fundamental_completo.empty:
        st.header("Sugestão de Alocação para Novo Aporte (Fronteira Markowitz + Quant Value)")
        carteira_markowitz = next((c for c in carteiras_comparativo_lista if c['Nome'].startswith('Otimizada Markowitz')), None)
        if carteira_markowitz and 'Pesos' in carteira_markowitz['Dados']:
            # Garante string simples na chave
            pesos_markowitz = {key_to_str(k): v for k, v in carteira_markowitz['Dados']['Pesos'].items()}
            quant_value = df_fundamental_completo['Quant_Value_Score']
            ativos_quant = quant_value.dropna().sort_values(ascending=False)
            ativos_filtrados = ativos_quant
            if top_n_quant_value > 0:
                ativos_filtrados = ativos_filtrados.head(top_n_quant_value)
            if min_quant_value > 0:
                ativos_filtrados = ativos_filtrados[ativos_filtrados >= min_quant_value]
            ativos_filtrados_index = [key_to_str(idx) for idx in ativos_filtrados.index]
            ativos_escolhidos = [a for a in pesos_markowitz.keys() if a in ativos_filtrados_index]
            if not ativos_escolhidos:
                st.warning("Nenhum ativo atende ao critério Quant Value escolhido. Usando todos os da Fronteira Markowitz.")
                ativos_escolhidos = list(pesos_markowitz.keys())
            pesos_filtrados = {a: pesos_markowitz[a] for a in ativos_escolhidos}
            soma_pesos = sum(pesos_filtrados.values())
            pesos_final = {a: p/soma_pesos for a, p in pesos_filtrados.items()}
            st.write(f"Ativos selecionados pelo Quant Value: {', '.join(map(str, ativos_escolhidos))}")
            compras_sugeridas, capital_excedente = sugerir_alocacao_novo_aporte(
                current_portfolio_composition_values=carteira_atual_composicao_valores,
                new_capital=novo_capital_input,
                target_portfolio_weights_decimal=pesos_final
            )
            if compras_sugeridas:
                df_compras = pd.DataFrame([
                    (key_to_str(k), v) for k, v in compras_sugeridas.items()
                ], columns=["Ativo", "Valor a Comprar"])
                st.dataframe(df_compras.sort_values("Valor a Comprar", ascending=False).style.format({"Valor a Comprar": "{:.2f}"}), use_container_width=True)
            else:
                st.write("Nenhuma compra sugerida (já alinhado ou aporte muito pequeno).")
            if capital_excedente > 0.01:
                st.write(f"Capital excedente após tentativa de alocação: R$ {capital_excedente:.2f}")
    st.success("Análise concluída!")
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise Avançada'.")


