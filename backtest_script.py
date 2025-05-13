import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Tentar importar funções do módulo de análise financeira
# Presume-se que financial_analyzer_enhanced_corrected.py está no mesmo diretório ou no PYTHONPATH
try:
    from financial_analyzer_enhanced_corrected import (
        obter_dados_historicos_yf,
        obter_dados_fundamentalistas_detalhados_br,
        calcular_piotroski_f_score_br,
        calcular_value_composite_score,
        RISK_FREE_RATE_DEFAULT  # Usar como default para taxa livre de risco
    )
    print("Módulo 'financial_analyzer_enhanced_corrected.py' carregado com sucesso.")
except ImportError as e:
    print(f"Erro ao importar 'financial_analyzer_enhanced_corrected.py': {e}. Certifique-se de que o arquivo está no diretório correto e todas as dependências estão instaladas.")
    # Em um cenário real, poderia sair do script ou usar implementações mock/padrão se o backtest puder prosseguir parcialmente.
    # Por agora, apenas imprimimos o erro.

# Definição padrão para VC_METRICS, similar ao Streamlit app
VC_METRICS_DEFAULT = [
    'trailingPE', 'priceToBook', 'enterpriseToEbitda', 
    'dividendYield', 'returnOnEquity', 'netMargin'
]
VC_METRIC_DIRECTIONS_DEFAULT = {
    'trailingPE': 'lower_is_better', 
    'priceToBook': 'lower_is_better', 
    'enterpriseToEbitda': 'lower_is_better',
    'dividendYield': 'higher_is_better', 
    'returnOnEquity': 'higher_is_better',
    'netMargin': 'higher_is_better',
    'forwardPE': 'lower_is_better',
    'marketCap': 'lower_is_better' # Adicionado para cobrir todas as opções do Streamlit
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script de Backtest para Estratégia Quantitativa.")

    # Parâmetros do Período de Backtest e Rebalanceamento
    parser.add_argument("--start_date_backtest", type=str, required=True, help="Data de início do backtest (YYYY-MM-DD).")
    parser.add_argument("--end_date_backtest", type=str, required=True, help="Data de fim do backtest (YYYY-MM-DD).")
    parser.add_argument("--initial_portfolio_value", type=float, default=100000.0, help="Valor inicial do portfólio.")
    parser.add_argument("--rebalance_frequency", type=str, default="monthly", choices=["monthly", "quarterly", "annually"], help="Frequência de rebalanceamento.")
    parser.add_argument("--historical_data_period_years", type=int, default=3, help="Número de anos de dados históricos para usar em cada rebalanceamento para calcular scores e retornos esperados.")

    # Parâmetros do Universo de Ativos e Filtros
    parser.add_argument("--universe_tickers", type=str, required=True, help="Lista de tickers do universo de investimento, separados por vírgula (ex: PETR4.SA,VALE3.SA).")
    parser.add_argument("--min_piotroski_score", type=int, default=0, help="Piotroski F-Score mínimo para inclusão de ativos (0-9). 0 para não filtrar.")
    parser.add_argument("--min_quant_value_score", type=float, default=0.0, help="Nota mínima do Quant Value Score para inclusão (0.0-1.0). 0.0 para não filtrar.")
    parser.add_argument("--top_n_quant_value", type=int, default=0, help="Selecionar os Top N ativos pelo Quant Value Score (0 para selecionar todos que passam no filtro de nota mínima).")
    parser.add_argument("--max_selected_assets", type=int, default=10, help="Número máximo de ativos a serem selecionados para a carteira em cada rebalanceamento.")

    # Parâmetros de Alocação
    parser.add_argument("--min_alloc_asset", type=float, default=0.05, help="Alocação mínima por ativo selecionado (ex: 0.05 para 5%%).")
    parser.add_argument("--max_alloc_asset", type=float, default=0.20, help="Alocação máxima por ativo selecionado (ex: 0.20 para 20%%).")
    # TODO: Adicionar estratégia de alocação (ex: 'equal_weight', 'value_weighted', 'min_variance')

    # Parâmetros para Value Composite Score
    parser.add_argument("--vc_metrics", type=str, nargs='+', default=VC_METRICS_DEFAULT, 
                        help=f"Métricas para o Value Composite Score. Default: {' '.join(VC_METRICS_DEFAULT)}.")

    # Outros Parâmetros
    parser.add_argument("--risk_free_rate", type=float, default=RISK_FREE_RATE_DEFAULT, help="Taxa livre de risco anual (ex: 0.02 para 2%%).")
    parser.add_argument("--output_dir", type=str, default="./backtest_results", help="Diretório para salvar os resultados do backtest.")

    args = parser.parse_args()
    
    # Validações e conversões adicionais
    try:
        datetime.strptime(args.start_date_backtest, "%Y-%m-%d")
        datetime.strptime(args.end_date_backtest, "%Y-%m-%d")
    except ValueError:
        parser.error("Datas devem estar no formato YYYY-MM-DD.")

    if datetime.strptime(args.start_date_backtest, "%Y-%m-%d") >= datetime.strptime(args.end_date_backtest, "%Y-%m-%d"):
        parser.error("A data de início do backtest deve ser anterior à data de fim.")
        
    args.universe_tickers = [ticker.strip().upper() for ticker in args.universe_tickers.split(',') if ticker.strip()]
    if not args.universe_tickers:
        parser.error("A lista de tickers do universo não pode estar vazia.")

    if not (0 <= args.min_alloc_asset <= 1 and 0 <= args.max_alloc_asset <= 1):
        parser.error("Alocação mínima e máxima por ativo deve estar entre 0 e 1.")
    if args.min_alloc_asset > args.max_alloc_asset:
        parser.error("Alocação mínima não pode ser maior que a alocação máxima.")

    # Criar diretório de saída se não existir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Diretório de saída criado: {args.output_dir}")

    return args

def get_rebalance_dates(start_date_str, end_date_str, frequency):
    """Gera datas de rebalanceamento com base na frequência."""
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    if frequency == "monthly":
        # Primeiro dia útil do mês
        rebalance_dates = pd.date_range(start_date, end_date, freq='BMS') 
    elif frequency == "quarterly":
        # Primeiro dia útil do trimestre
        rebalance_dates = pd.date_range(start_date, end_date, freq='BQS') 
    elif frequency == "annually":
        # Primeiro dia útil do ano
        rebalance_dates = pd.date_range(start_date, end_date, freq='BYS') 
    else:
        raise ValueError(f"Frequência de rebalanceamento desconhecida: {frequency}")
    
    # Garante que as datas de rebalanceamento estão dentro do período de backtest
    # e que a primeira data de rebalanceamento não é antes da data de início do backtest.
    rebalance_dates = rebalance_dates[rebalance_dates >= start_date]
    rebalance_dates = rebalance_dates[rebalance_dates <= end_date]
    
    if rebalance_dates.empty:
        print(f"Nenhuma data de rebalanceamento gerada para o período {start_date_str} a {end_date_str} com frequência {frequency}.")
        
    return rebalance_dates.to_list()


def run_backtest(args):
    print("Iniciando o processo de backtest...")
    print(f"Parâmetros recebidos: {args}")

    vc_metrics_config_backtest = {metric: VC_METRIC_DIRECTIONS_DEFAULT[metric] 
                                  for metric in args.vc_metrics 
                                  if metric in VC_METRIC_DIRECTIONS_DEFAULT}
    if len(vc_metrics_config_backtest) != len(args.vc_metrics):
        print("Aviso: Algumas vc_metrics fornecidas não são reconhecidas e serão ignoradas.")

    rebalance_dates = get_rebalance_dates(args.start_date_backtest, args.end_date_backtest, args.rebalance_frequency)
    if not rebalance_dates:
        print("Nenhuma data de rebalanceamento. Encerrando o backtest.")
        return

    print(f"Datas de rebalanceamento ({len(rebalance_dates)}): {rebalance_dates}")

    portfolio_history = [] # Lista para armazenar (data, valor_portfolio, composicao_atual)
    current_portfolio_value = args.initial_portfolio_value
    current_holdings = {} # Dicionário {ticker: {'shares': X, 'price_at_buy': Y, 'weight': Z}}

    all_price_data_for_simulation = {} # Cache para dados de preços usados na simulação

    # Loop principal de rebalanceamento
    for i, current_rebalance_date in enumerate(rebalance_dates):
        print(f"\n--- Rebalanceamento em: {current_rebalance_date.strftime('%Y-%m-%d')} ---")
        
        # 1. Determinar período para coleta de dados fundamentalistas e históricos para scores
        # Usar dados até o dia *anterior* ao rebalanceamento para evitar lookahead bias.
        data_collection_end_date = current_rebalance_date - pd.Timedelta(days=1)
        data_collection_start_date = data_collection_end_date - pd.Timedelta(days=args.historical_data_period_years * 365)
        
        print(f"Período de coleta de dados para scores: {data_collection_start_date.strftime('%Y-%m-%d')} a {data_collection_end_date.strftime('%Y-%m-%d')}")

        # 2. Obter dados fundamentalistas e históricos para o universo de ativos
        # Idealmente, dados fundamentalistas seriam point-in-time. A função atual pode não suportar isso.
        # obter_dados_fundamentalistas_detalhados_br provavelmente pega os dados mais recentes.
        # Para um backtest mais preciso, isso precisaria de uma fonte de dados históricos fundamentalistas.
        # Por ora, usaremos como está, cientes da limitação.
        df_fundamental = obter_dados_fundamentalistas_detalhados_br(args.universe_tickers)
        if df_fundamental.empty:
            print("Não foi possível obter dados fundamentalistas. Pulando este rebalanceamento.")
            # Adicionar o valor do portfólio atual ao histórico se houver holdings
            if current_holdings:
                 portfolio_history.append({
                    "date": current_rebalance_date,
                    "portfolio_value": current_portfolio_value, # Valor antes do rebalanceamento
                    "holdings": current_holdings.copy() 
                })
            continue
        
        df_fundamental.set_index('ticker', inplace=True, drop=False)

        # 3. Calcular Scores
        # Piotroski F-Score
        piotroski_results = df_fundamental.apply(
            lambda row: calcular_piotroski_f_score_br(row, verbose=True)[:2], # Pega score e detalhes, alinhado com Streamlit
            axis=1
        )
        df_fundamental["Piotroski_F_Score"] = [res[0] for res in piotroski_results]
        # df_fundamental["Piotroski_F_Detalhes"] = [res[1] for res in piotroski_results]

        # Quant Value Score
        df_fundamental['Quant_Value_Score'] = calcular_value_composite_score(df_fundamental, vc_metrics_config_backtest)
        
        # 4. Selecionar Ativos
        # Aplicar filtros
        df_elegiveis = df_fundamental.copy()
        if args.min_piotroski_score > 0:
            df_elegiveis = df_elegiveis[df_elegiveis["Piotroski_F_Score"] >= args.min_piotroski_score]
        if args.min_quant_value_score > 0.0:
            df_elegiveis = df_elegiveis[df_elegiveis["Quant_Value_Score"] >= args.min_quant_value_score]
        
        # Ordenar por Quant Value Score (maior primeiro) e selecionar Top N se aplicável
        df_elegiveis = df_elegiveis.sort_values(by="Quant_Value_Score", ascending=False)
        if args.top_n_quant_value > 0:
            df_elegiveis = df_elegiveis.head(args.top_n_quant_value)
        
        # Limitar pelo número máximo de ativos
        selected_tickers = df_elegiveis.head(args.max_selected_assets)['ticker'].tolist()

        if not selected_tickers:
            print("Nenhum ativo selecionado após filtros. Mantendo portfólio anterior (se houver) ou caixa.")
            # Se não há ativos selecionados, o portfólio fica em caixa ou mantém posições anteriores até o próximo rebalanceamento.
            # Para simplificar, vamos assumir que o valor do portfólio não muda se não houver rebalanceamento de ativos.
            # A simulação da evolução do valor do portfólio (próxima etapa) cuidará disso.
        else:
            print(f"Ativos selecionados para {current_rebalance_date.strftime('%Y-%m-%d')}: {selected_tickers}")

            # 5. Calcular Pesos Alvo (Estratégia de Peso Igual por enquanto)
            num_selected = len(selected_tickers)
            target_weights = {}
            if num_selected > 0:
                base_weight = 1.0 / num_selected
                # Ajustar para min/max allocation - isso pode ser complexo se os limites forem rígidos
                # Por ora, uma alocação igual simples, mas respeitando os limites se possível.
                # Se base_weight < min_alloc_asset ou > max_alloc_asset, a lógica precisa ser mais robusta.
                # Exemplo: se 1/N < min_alloc, não podemos ter N ativos. Reduzir N.
                # Se 1/N > max_alloc, ok, mas se quisermos usar max_alloc, o restante é distribuído.
                
                # Lógica simplificada: se o peso igual estiver fora dos limites, ajustamos o número de ativos
                # ou aplicamos os limites e normalizamos. Por agora, vamos usar peso igual e depois aplicar.
                
                # Verificar se é possível atender min_alloc com o número de ativos
                if args.min_alloc_asset * num_selected > 1.0:
                    print(f"Aviso: Com {num_selected} ativos, a alocação mínima de {args.min_alloc_asset*100:.1f}% por ativo excede 100%. Ajustando número de ativos.")
                    num_selected = int(1.0 / args.min_alloc_asset)
                    selected_tickers = selected_tickers[:num_selected]
                    print(f"Novos ativos selecionados ({num_selected}): {selected_tickers}")
                
                if not selected_tickers: # Se após ajuste não sobrar ativos
                    print("Nenhum ativo restante após ajuste de alocação mínima. Mantendo caixa.")
                    current_holdings = {} # Zera as posições
                else:
                    weight_per_asset = 1.0 / len(selected_tickers)
                    # Aplicar clamp para min/max
                    final_weights_raw = {ticker: np.clip(weight_per_asset, args.min_alloc_asset, args.max_alloc_asset) for ticker in selected_tickers}
                    
                    # Normalizar pesos se a soma não for 1 (devido ao clamp)
                    sum_weights = sum(final_weights_raw.values())
                    target_weights = {ticker: w / sum_weights for ticker, w in final_weights_raw.items()}

                    print(f"Pesos alvo calculados: {target_weights}")

                    # 6. Simular Transações (Rebalanceamento)
                    # Obter preços na data de rebalanceamento para calcular as quantidades de ações
                    # Idealmente, usar preços de abertura do dia de rebalanceamento ou fechamento do dia anterior.
                    # Para simplificar, vamos buscar o preço de fechamento na data de rebalanceamento.
                    # Se a data de rebalanceamento for um não-dia útil, yfinance pegará o próximo dia útil.
                    
                    new_holdings = {}
                    if target_weights:
                        prices_on_rebalance_date_df = obter_dados_historicos_yf(selected_tickers, 
                                                                                current_rebalance_date.strftime("%Y-%m-%d"), 
                                                                                (current_rebalance_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
                        # column=\'Adj Close\'                       
                        if prices_on_rebalance_date_df.empty or prices_on_rebalance_date_df.isnull().all().all():
                            print(f"Não foi possível obter preços para os ativos selecionados em {current_rebalance_date.strftime('%Y-%m-%d')}. Mantendo portfólio anterior ou caixa.")
                            # Mantém current_holdings como estava antes deste rebalanceamento falho
                        else:
                            # Pegar o primeiro preço válido disponível (geralmente o da própria data de rebalanceamento)
                            prices_on_rebalance_date = prices_on_rebalance_date_df.ffill().bfill().iloc[0]
                            
                            prices_available_for_selected = prices_on_rebalance_date.reindex(selected_tickers)
                            missing_prices = prices_available_for_selected[prices_available_for_selected.isnull()].index.tolist()
                            if missing_prices:
                                print(f"Aviso: Preços não encontrados para {missing_prices} em {current_rebalance_date.strftime('%Y-%m-%d')}. Esses ativos não serão incluídos neste rebalanceamento.")
                                # Recalcular pesos sem esses ativos
                                selected_tickers = [t for t in selected_tickers if t not in missing_prices]
                                if selected_tickers:
                                    weight_per_asset = 1.0 / len(selected_tickers)
                                    final_weights_raw = {ticker: np.clip(weight_per_asset, args.min_alloc_asset, args.max_alloc_asset) for ticker in selected_tickers}
                                    sum_weights = sum(final_weights_raw.values())
                                    target_weights = {ticker: w / sum_weights for ticker, w in final_weights_raw.items()}
                                else:
                                    target_weights = {}
                                    print("Nenhum ativo restante após remoção por falta de preço. Mantendo caixa.")

                            if target_weights:
                                current_holdings = {} # Zera as posições antigas para rebalancear
                                for ticker, weight in target_weights.items():
                                    if ticker in prices_on_rebalance_date and not pd.isnull(prices_on_rebalance_date[ticker]):
                                        price = prices_on_rebalance_date[ticker]
                                        amount_to_invest = current_portfolio_value * weight
                                        shares = amount_to_invest / price
                                        new_holdings[ticker] = {'shares': shares, 'price_at_buy': price, 'weight': weight}
                                    else:
                                        print(f"Preço para {ticker} ainda ausente, não será incluído.")
                                current_holdings = new_holdings
                            else:
                                current_holdings = {} # Zera se não há target_weights válidos
                    else: # Nenhum ativo selecionado ou problema nos pesos
                        current_holdings = {} # Zera as posições
        
        # 7. Simular Evolução do Portfólio até o Próximo Rebalanceamento (ou fim do backtest)
        start_sim_period = current_rebalance_date
        if i + 1 < len(rebalance_dates):
            end_sim_period = rebalance_dates[i+1] - pd.Timedelta(days=1) # Um dia antes do próximo rebalanceamento
        else:
            end_sim_period = pd.to_datetime(args.end_date_backtest)
        
        print(f"Simulando valor do portfólio de {start_sim_period.strftime('%Y-%m-%d')} a {end_sim_period.strftime('%Y-%m-%d')}")

        if current_holdings:
            tickers_in_portfolio = list(current_holdings.keys())
            # Buscar dados de preços para o período de simulação
            # Otimização: buscar apenas uma vez e armazenar em cache se possível
            # Por simplicidade, buscamos a cada período de simulação.
            # Adicionar uma pequena folga na data final para garantir que pegamos o último dia.
            daily_prices_df = obter_dados_historicos_yf(tickers_in_portfolio, 
                                                        start_sim_period.strftime("%Y-%m-%d"), 
                                                        (end_sim_period + pd.Timedelta(days=3)).strftime("%Y-%m-%d")) 
            # column=\'Adj Close\'
            daily_prices_df = daily_prices_df.ffill() # Preencher NaNs com o último valor válido
            
            # Iterar pelos dias no período de simulação
            for sim_date in pd.date_range(start_sim_period, end_sim_period, freq='B'): # Dias úteis
                if sim_date not in daily_prices_df.index:
                    # Se não houver dados para este dia (ex: feriado não coberto por 'B'), usar o valor do dia anterior
                    if portfolio_history and portfolio_history[-1]['date'] < sim_date:
                        # Mantém o valor do portfólio do último dia registrado se for anterior
                        last_value = portfolio_history[-1]['portfolio_value']
                        portfolio_history.append({
                            "date": sim_date,
                            "portfolio_value": last_value,
                            "holdings": current_holdings.copy() # Holdings não mudam entre rebalanceamentos
                        })
                    elif not portfolio_history: # Caso seja o primeiro dia e não haja dados
                         portfolio_history.append({
                            "date": sim_date,
                            "portfolio_value": current_portfolio_value, # Valor antes de qualquer mudança
                            "holdings": current_holdings.copy()
                        })
                    continue

                current_day_value = 0
                temp_holdings_details_for_log = {}
                valid_prices_for_day = True
                for ticker, data in current_holdings.items():
                    if ticker in daily_prices_df.columns and sim_date in daily_prices_df.index and not pd.isnull(daily_prices_df.loc[sim_date, ticker]):
                        current_price = daily_prices_df.loc[sim_date, ticker]
                        current_day_value += data['shares'] * current_price
                        temp_holdings_details_for_log[ticker] = {'shares': data['shares'], 'current_price': current_price, 'value': data['shares'] * current_price}
                    else:
                        # Se o preço de um ativo estiver faltando, isso é um problema.
                        # Poderia usar o último preço conhecido ou zerar o valor do ativo.
                        # Por simplicidade, se um preço faltar, o valor do portfólio pode ficar impreciso.
                        # Vamos tentar usar o último valor conhecido do ativo se disponível, ou manter o valor do portfólio do dia anterior.
                        print(f"Aviso: Preço para {ticker} não encontrado em {sim_date.strftime('%Y-%m-%d')}. Tentando usar último valor conhecido do ativo.")
                        if portfolio_history and ticker in portfolio_history[-1]['holdings'] and 'current_price' in portfolio_history[-1]['holdings'][ticker]:
                            last_known_price = portfolio_history[-1]['holdings'][ticker]['current_price']
                            current_day_value += data['shares'] * last_known_price
                            temp_holdings_details_for_log[ticker] = {'shares': data['shares'], 'current_price': last_known_price, 'value': data['shares'] * last_known_price}
                        else: # Não há como calcular, portfólio pode estar incorreto
                            print(f"Não foi possível encontrar preço para {ticker} em {sim_date.strftime('%Y-%m-%d')}, nem último conhecido. Valor do portfólio pode ser afetado.")
                            valid_prices_for_day = False
                            break # Interrompe cálculo do valor do dia
                
                if valid_prices_for_day:
                    current_portfolio_value = current_day_value
                elif portfolio_history: # Se preços não válidos, usa o valor do dia anterior
                    current_portfolio_value = portfolio_history[-1]['portfolio_value']
                # else: current_portfolio_value permanece o valor do rebalanceamento (se for o primeiro dia)

                portfolio_history.append({
                    "date": sim_date,
                    "portfolio_value": current_portfolio_value,
                    "holdings": temp_holdings_details_for_log if valid_prices_for_day else current_holdings.copy() # Log detalhado se possível
                })
        else: # Portfólio está em caixa (sem holdings)
            for sim_date in pd.date_range(start_sim_period, end_sim_period, freq='B'):
                portfolio_history.append({
                    "date": sim_date,
                    "portfolio_value": current_portfolio_value, # Valor em caixa não muda
                    "holdings": {}
                })
        
        print(f"Valor do portfólio ao final de {end_sim_period.strftime('%Y-%m-%d')}: {current_portfolio_value:.2f}")

    # 8. Salvar resultados
    if portfolio_history:
        df_portfolio_history = pd.DataFrame(portfolio_history)
        df_portfolio_history.set_index('date', inplace=True)
        
        output_file = os.path.join(args.output_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_portfolio_history.to_csv(output_file)
        print(f"\nHistórico do portfólio salvo em: {output_file}")

        # TODO: Calcular e exibir métricas de backtest (Sharpe, Drawdown, etc.)
        # calculate_backtest_metrics(df_portfolio_history, args.risk_free_rate)
    else:
        print("Nenhum histórico de portfólio gerado.")

    print("Processo de backtest concluído.")


if __name__ == "__main__":
    args = parse_arguments()
    run_backtest(args)


