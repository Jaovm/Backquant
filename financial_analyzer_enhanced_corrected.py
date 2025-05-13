#!/usr/bin/python3.11
# financial_analyzer_enhanced.py

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
from arch import arch_model
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# --- Configuration & Constants ---
DATA_DIR = "/home/ubuntu/collected_data"
RISK_FREE_RATE_DEFAULT = 0.02 # Annual risk-free rate
MIN_YEARS_DATA = 3 # Minimum years of data for some calculations

# --- Data Loading and Preprocessing ---

def load_historical_data_from_csv(ticker):
    """Loads historical price data from a CSV file in DATA_DIR."""
    safe_ticker_name = ticker.replace('^', 'INDEX_')
    file_path = os.path.join(DATA_DIR, f"{safe_ticker_name}_chart_data.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            if 'Adj Close' in df.columns and pd.api.types.is_numeric_dtype(df['Adj Close']):
                df.dropna(subset=['Adj Close'], inplace=True)
                return df[['Adj Close']]
            else:
                print(f"Warning: 'Adj Close' not found or not numeric in {file_path} for {ticker}.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading or processing CSV {file_path} for {ticker}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def obter_dados_historicos_yf(ativos, start_date_str, end_date_str=None):
    """Obtém retornos diários dos ativos, prioritizing CSV then Yahoo Finance API."""
    if end_date_str is None:
        end_date_str = datetime.today().strftime('%Y-%m-%d')
    
    all_adj_close = pd.DataFrame()
    for ativo in ativos:
        df_ativo = load_historical_data_from_csv(ativo)
        if df_ativo.empty:
            print(f"Fetching {ativo} from yfinance API as CSV not found or invalid.")
            try:
                data = yf.download(ativo, start=start_date_str, end=end_date_str, progress=False)
                if not data.empty and 'Close' in data.columns:
                    df_ativo = data[['Close']].copy()
                    df_ativo.rename(columns={'Close': 'Adj Close'}, inplace=True)
                    df_ativo.dropna(inplace=True)
                else:
                    print(f"Warning: No 'Adj Close' data from yfinance API for {ativo}.")
                    continue
            except Exception as e:
                print(f"Error fetching {ativo} from yfinance API: {e}")
                continue
        
        if not df_ativo.empty:
            df_ativo = df_ativo[(df_ativo.index >= pd.to_datetime(start_date_str)) & (df_ativo.index <= pd.to_datetime(end_date_str))]
            df_ativo.rename(columns={'Adj Close': ativo}, inplace=True)
            if all_adj_close.empty:
                all_adj_close = df_ativo
            else:
                all_adj_close = all_adj_close.join(df_ativo, how='outer')
        else:
            print(f"Warning: No data loaded for {ativo}.")

    if all_adj_close.empty:
        return pd.DataFrame()
        
    df_retornos = all_adj_close.pct_change().dropna(how='all')
    return df_retornos

def load_insights_data_from_json(ticker):
    """Loads insights data from a JSON file in DATA_DIR."""
    safe_ticker_name = ticker.replace('^', 'INDEX_')
    file_path = os.path.join(DATA_DIR, f"{safe_ticker_name}_insights_data.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON {file_path} for {ticker}: {e}")
    return None

def get_yfinance_ticker_info(ticker_symbol):
    """Safely fetches yfinance.Ticker object and its info."""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        return ticker_obj.info, ticker_obj
    except Exception as e:
        print(f"Error fetching yfinance.Ticker info for {ticker_symbol}: {e}")
        return None, None

# --- Fundamental Metrics Helper ---
def _get_numeric_value(row, field_names, default=np.nan):
    """
    Tries to get a value from a row using a list of possible field names.
    Converts the value to numeric. Returns default (np.nan) if not found or conversion fails.
    Input `row` can be a Pandas Series or a dictionary.
    `field_names` is a list of strings.
    """
    val = default
    if isinstance(field_names, str):
        field_names = [field_names] # Ensure it's a list

    for name in field_names:
        try:
            if isinstance(row, pd.Series):
                if name in row.index and pd.notna(row[name]):
                    val = row[name]
                    break
            elif isinstance(row, dict): # Handle if row is a dict
                if name in row and pd.notna(row.get(name)):
                    val = row.get(name)
                    break
        except Exception:
            continue # Ignore errors during lookup, try next name
            
    return pd.to_numeric(val, errors='coerce')

# --- Fundamental Metrics --- 
def obter_dados_fundamentalistas_detalhados_br(ativos):
    """
    Obtém dados fundamentalistas detalhados para ativos da B3, usando yfinance,
    e complementando com APIs nacionais (exemplo: Brapi) se necessário.
    Retorna DataFrame padronizado para análise fundamentalista avançada.
    """
    dados_fund = {}
    for ativo in ativos:
        temp_data = {'ticker': ativo}
        info, yf_ticker_obj = get_yfinance_ticker_info(ativo)
        
        if info:
            # Prioritize yfinance.info for these general fields
            temp_data['marketCap'] = info.get('marketCap')
            temp_data['sharesOutstanding'] = info.get('sharesOutstanding') # Current shares
            temp_data['dividendYield'] = info.get('dividendYield')
            temp_data['trailingPE'] = info.get('trailingPE')
            temp_data['priceToBook'] = info.get('priceToBook')
            temp_data['returnOnEquity'] = info.get('returnOnEquity')
            temp_data['totalRevenue'] = info.get('totalRevenue') # TTM Revenue from info
            temp_data['operatingCashflow'] = info.get('operatingCashflow') # TTM CFO from info
            temp_data['totalDebt'] = info.get('totalDebt')
            temp_data['grossMargins'] = info.get('grossMargins') # TTM Gross Margin from info
            temp_data['netIncomeToCommon'] = info.get('netIncomeToCommon') # TTM Net Income from info
            temp_data['sector'] = info.get('sector')
            temp_data['industry'] = info.get('industry')

        if yf_ticker_obj:
            try:
                financials = yf_ticker_obj.financials
                balance_sheet = yf_ticker_obj.balance_sheet
                cashflow = yf_ticker_obj.cashflow
                
                # Current year data (most recent column, usually index 0)
                if not financials.empty and len(financials.columns) > 0:
                    col_curr = financials.columns[0]
                    temp_data['lucro_liquido_atual'] = financials.loc['Net Income', col_curr] if 'Net Income' in financials.index else temp_data.get('netIncomeToCommon')
                    temp_data['lucro_bruto_atual'] = financials.loc['Gross Profit', col_curr] if 'Gross Profit' in financials.index else None
                    temp_data['receita_liquida_atual'] = financials.loc['Total Revenue', col_curr] if 'Total Revenue' in financials.index else temp_data.get('totalRevenue')
                
                if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                    col_curr = balance_sheet.columns[0]
                    temp_data['ativos_totais_atual'] = balance_sheet.loc['Total Assets', col_curr] if 'Total Assets' in balance_sheet.index else None
                    temp_data['divida_lp_atual'] = balance_sheet.loc['Long Term Debt', col_curr] if 'Long Term Debt' in balance_sheet.index else None
                    temp_data['ativos_circulantes_atual'] = balance_sheet.loc['Total Current Assets', col_curr] if 'Total Current Assets' in balance_sheet.index else None
                    temp_data['passivos_circulantes_atual'] = balance_sheet.loc['Total Current Liabilities', col_curr] if 'Total Current Liabilities' in balance_sheet.index else None
                    temp_data['patrimonio_liquido_atual'] = balance_sheet.loc['Total Stockholder Equity', col_curr] if 'Total Stockholder Equity' in balance_sheet.index else None
                    # For shares outstanding previous year, we might need to look at historical balance sheets if available
                    # yfinance 'sharesOutstanding' is usually the most current. For a truly 'previous year shares', it's complex.
                    # We'll rely on a field 'sharesOutstanding_prev' if provided by another source or manually added later.
                    # For now, let's try to get common shares issued from balance sheet as a proxy if available
                    if 'Ordinary Shares Number' in balance_sheet.index:
                         temp_data['sharesOutstanding_from_bs_curr'] = balance_sheet.loc['Ordinary Shares Number', col_curr]
                    elif 'Share Issued' in balance_sheet.index:
                         temp_data['sharesOutstanding_from_bs_curr'] = balance_sheet.loc['Share Issued', col_curr]

                if not cashflow.empty and len(cashflow.columns) > 0:
                    col_curr = cashflow.columns[0]
                    temp_data['cfo_atual'] = cashflow.loc['Total Cash From Operating Activities', col_curr] if 'Total Cash From Operating Activities' in cashflow.index else temp_data.get('operatingCashflow')

                # Previous year data (second most recent column, usually index 1)
                if len(financials.columns) >= 2:
                    col_prev = financials.columns[1]
                    temp_data['lucro_liquido_anterior'] = financials.loc['Net Income', col_prev] if 'Net Income' in financials.index else None
                    temp_data['lucro_bruto_anterior'] = financials.loc['Gross Profit', col_prev] if 'Gross Profit' in financials.index else None
                    temp_data['receita_liquida_anterior'] = financials.loc['Total Revenue', col_prev] if 'Total Revenue' in financials.index else None
                
                if len(balance_sheet.columns) >= 2:
                    col_prev = balance_sheet.columns[1]
                    temp_data['ativos_totais_anterior'] = balance_sheet.loc['Total Assets', col_prev] if 'Total Assets' in balance_sheet.index else None
                    temp_data['divida_lp_anterior'] = balance_sheet.loc['Long Term Debt', col_prev] if 'Long Term Debt' in balance_sheet.index else None
                    temp_data['ativos_circulantes_anterior'] = balance_sheet.loc['Total Current Assets', col_prev] if 'Total Current Assets' in balance_sheet.index else None
                    temp_data['passivos_circulantes_anterior'] = balance_sheet.loc['Total Current Liabilities', col_prev] if 'Total Current Liabilities' in balance_sheet.index else None
                    temp_data['patrimonio_liquido_anterior'] = balance_sheet.loc['Total Stockholder Equity', col_prev] if 'Total Stockholder Equity' in balance_sheet.index else None
                    if 'Ordinary Shares Number' in balance_sheet.index:
                         temp_data['sharesOutstanding_from_bs_prev'] = balance_sheet.loc['Ordinary Shares Number', col_prev]
                    elif 'Share Issued' in balance_sheet.index:
                         temp_data['sharesOutstanding_from_bs_prev'] = balance_sheet.loc['Share Issued', col_prev]

            except Exception as e:
                print(f"Erro ao buscar dados detalhados de yfinance para {ativo}: {e}")

        # Complementar com Brapi (exemplo)
        # A função preencher_campos_faltantes_brapi(row) mencionada não existe no script original.
        # Esta seção simula uma tentativa de preenchimento.
        try:
            # Idealmente, o token da API não deveria estar hardcoded.
            url_brapi = f"https://brapi.dev/api/quote/{ativo}?token=2D29LijXrSGRJAQ7De5bUh&fundamental=true"
            # Substitua YOUR_BRAPI_TOKEN_HERE pelo seu token real ou gerencie-o via config/env var.
            r = requests.get(url_brapi, timeout=5)
            if r.status_code == 200:
                brapi_results = r.json().get('results')
                if brapi_results and len(brapi_results) > 0:
                    # Brapi data often refers to the most recent available, not necessarily TTM or specific fiscal years.
                    # Careful mapping is needed.
                    # Example: if yfinance 'operatingCashflow' is missing, try Brapi's equivalent if available.
                    # This part needs a more robust mapping based on Brapi's actual field names and definitions.
                    # For now, just adding a few examples that were in the original script.
                    brapi_data = brapi_results[0] # Assuming first result is the one
                    if temp_data.get('roic') is None: temp_data['roic'] = brapi_data.get('roic')
                    if temp_data.get('ebit_margin') is None: temp_data['ebit_margin'] = brapi_data.get('ebitMargin')
                    # ... (add other relevant fields from Brapi if needed and map them correctly)
        except Exception as e:
            print(f"Falha ao buscar dados complementares da Brapi para {ativo}: {e}")

        dados_fund[ativo] = temp_data

    df_fund = pd.DataFrame.from_dict(dados_fund, orient='index')
    # Conversão numérica robusta para todas as colunas após a coleta
    for col in df_fund.columns:
        if col != 'ticker' and col != 'sector' and col != 'industry': # Evitar converter colunas de texto
            df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')
    return df_fund

def calcular_piotroski_f_score_br(row, verbose=False):
    """
    Calcula o Piotroski F-Score e retorna score e detalhamento dos critérios para empresas brasileiras.
    Revisado para maior robustez e clareza, conforme solicitado.
    Parâmetros:
        row: linha do dataframe de fundamentos (Pandas Series).
        verbose: se True, retorna score, dict de critérios e dict de valores intermediários.
    """
    criterios = {}
    debug_valores = {}

    # --- Rentabilidade ---
    # 1. Lucro Líquido positivo (ROA_SCORE)
    lucro_liquido_atual = _get_numeric_value(row, ['lucro_liquido_atual', 'netIncomeToCommon', 'netIncome', 'NetIncome_curr', 'NI_curr'])
    debug_valores['lucro_liquido_atual'] = lucro_liquido_atual
    criterios['lucro_liquido_positivo'] = 1 if pd.notna(lucro_liquido_atual) and lucro_liquido_atual > 0 else 0

    # 2. Fluxo de Caixa Operacional positivo (CFO_SCORE)
    cfo_atual = _get_numeric_value(row, ['cfo_atual', 'operatingCashflow', 'totalCashFromOperatingActivities', 'CFO_curr'])
    debug_valores['cfo_atual'] = cfo_atual
    criterios['cfo_positivo'] = 1 if pd.notna(cfo_atual) and cfo_atual > 0 else 0

    # 3. ROA crescente (DELTA_ROA_SCORE)
    ativos_totais_atual = _get_numeric_value(row, ['ativos_totais_atual', 'totalAssets', 'TotalAssets_curr'])
    ativos_totais_anterior = _get_numeric_value(row, ['ativos_totais_anterior', 'TotalAssets_prev'])
    lucro_liquido_anterior = _get_numeric_value(row, ['lucro_liquido_anterior', 'NetIncome_prev', 'NI_prev'])
    debug_valores.update({'ativos_totais_atual': ativos_totais_atual, 'ativos_totais_anterior': ativos_totais_anterior, 'lucro_liquido_anterior': lucro_liquido_anterior})

    roa_atual, roa_anterior = np.nan, np.nan
    try:
        if pd.notna(lucro_liquido_atual) and pd.notna(ativos_totais_atual) and ativos_totais_atual != 0:
            roa_atual = lucro_liquido_atual / ativos_totais_atual
    except Exception: roa_atual = np.nan
    try:
        if pd.notna(lucro_liquido_anterior) and pd.notna(ativos_totais_anterior) and ativos_totais_anterior != 0:
            roa_anterior = lucro_liquido_anterior / ativos_totais_anterior
    except Exception: roa_anterior = np.nan
    debug_valores.update({'roa_atual': roa_atual, 'roa_anterior': roa_anterior})
    criterios['roa_crescente'] = 1 if pd.notna(roa_atual) and pd.notna(roa_anterior) and roa_atual > roa_anterior else 0

    # 4. Qualidade do Lucro: CFO > Lucro Líquido (ACCRUAL_SCORE)
    criterios['cfo_maior_que_lucro'] = 1 if pd.notna(cfo_atual) and pd.notna(lucro_liquido_atual) and cfo_atual > lucro_liquido_atual else 0

    # --- Alavancagem, Liquidez e Fontes de Financiamento ---
    # 5. Queda da dívida de longo prazo (DELTA_LEVER_SCORE)
    # (Dívida LP / Ativos Totais)_atual < (Dívida LP / Ativos Totais)_anterior
    divida_lp_atual = _get_numeric_value(row, ['divida_lp_atual', 'longTermDebt', 'LongTermDebt_curr'])
    divida_lp_anterior = _get_numeric_value(row, ['divida_lp_anterior', 'LongTermDebt_prev'])
    debug_valores.update({'divida_lp_atual': divida_lp_atual, 'divida_lp_anterior': divida_lp_anterior})

    leverage_atual, leverage_anterior = np.nan, np.nan
    try:
        if pd.notna(divida_lp_atual) and pd.notna(ativos_totais_atual) and ativos_totais_atual != 0:
            leverage_atual = divida_lp_atual / ativos_totais_atual
    except Exception: leverage_atual = np.nan
    try:
        if pd.notna(divida_lp_anterior) and pd.notna(ativos_totais_anterior) and ativos_totais_anterior != 0:
            leverage_anterior = divida_lp_anterior / ativos_totais_anterior
    except Exception: leverage_anterior = np.nan
    debug_valores.update({'leverage_atual': leverage_atual, 'leverage_anterior': leverage_anterior})
    
    criterios['queda_divida_lp'] = 0
    if pd.notna(leverage_atual) and pd.notna(leverage_anterior):
        if leverage_atual < leverage_anterior:
            criterios['queda_divida_lp'] = 1
    elif pd.notna(leverage_atual) and leverage_atual == 0: # No long-term debt now
        criterios['queda_divida_lp'] = 1 # Considered positive if current leverage is zero
    elif pd.isna(leverage_atual) and pd.notna(leverage_anterior) and leverage_anterior == 0: # No debt before, no info now
        criterios['queda_divida_lp'] = 1 # If it was zero and now is NaN, assume it's still zero or better

    # 6. Aumento do índice de liquidez corrente (DELTA_LIQUID_SCORE)
    ativos_circulantes_atual = _get_numeric_value(row, ['ativos_circulantes_atual', 'totalCurrentAssets', 'CurrentAssets_curr'])
    passivos_circulantes_atual = _get_numeric_value(row, ['passivos_circulantes_atual', 'totalCurrentLiabilities', 'CurrentLiab_curr'])
    ativos_circulantes_anterior = _get_numeric_value(row, ['ativos_circulantes_anterior', 'CurrentAssets_prev'])
    passivos_circulantes_anterior = _get_numeric_value(row, ['passivos_circulantes_anterior', 'CurrentLiab_prev'])
    debug_valores.update({'ativos_circ_atual': ativos_circulantes_atual, 'passivos_circ_atual': passivos_circulantes_atual, 
                           'ativos_circ_ant': ativos_circulantes_anterior, 'passivos_circ_ant': passivos_circulantes_anterior})

    liquidez_corrente_atual, liquidez_corrente_anterior = np.nan, np.nan
    try:
        if pd.notna(ativos_circulantes_atual) and pd.notna(passivos_circulantes_atual) and passivos_circulantes_atual != 0:
            liquidez_corrente_atual = ativos_circulantes_atual / passivos_circulantes_atual
    except Exception: liquidez_corrente_atual = np.nan
    try:
        if pd.notna(ativos_circulantes_anterior) and pd.notna(passivos_circulantes_anterior) and passivos_circulantes_anterior != 0:
            liquidez_corrente_anterior = ativos_circulantes_anterior / passivos_circulantes_anterior
    except Exception: liquidez_corrente_anterior = np.nan
    debug_valores.update({'liquidez_corr_atual': liquidez_corrente_atual, 'liquidez_corr_anterior': liquidez_corrente_anterior})
    criterios['aumento_liquidez_corrente'] = 1 if pd.notna(liquidez_corrente_atual) and pd.notna(liquidez_corrente_anterior) and liquidez_corrente_atual > liquidez_corrente_anterior else 0

    # 7. Não emissão de ações (EQ_OFFER_SCORE)
    # Ideal: Common Shares Outstanding. yfinance `sharesOutstanding` is usually TTM/current.
    # `sharesOutstanding_from_bs_curr` and `sharesOutstanding_from_bs_prev` are proxies from balance sheet.
    acoes_emitidas_atual = _get_numeric_value(row, ['sharesOutstanding', 'acoes_emitidas_atual', 'sharesOutstanding_from_bs_curr'])
    acoes_emitidas_anterior = _get_numeric_value(row, ['sharesOutstanding_prev', 'acoes_emitidas_anterior', 'sharesOutstanding_from_bs_prev'])
    debug_valores.update({'acoes_emitidas_atual': acoes_emitidas_atual, 'acoes_emitidas_anterior': acoes_emitidas_anterior})

    criterios['nao_emitiu_acoes'] = 0
    if pd.notna(acoes_emitidas_atual) and pd.notna(acoes_emitidas_anterior):
        if acoes_emitidas_atual <= acoes_emitidas_anterior: # No dilution or buyback
            criterios['nao_emitiu_acoes'] = 1
    # If previous is NaN but current is available, we cannot make a comparison for change.
    # Piotroski implies a comparison, so 0 points if prior year data is missing.

    # --- Eficiência Operacional ---
    # 8. Aumento da margem bruta (DELTA_MARGIN_SCORE)
    lucro_bruto_atual = _get_numeric_value(row, ['lucro_bruto_atual', 'grossProfit', 'GrossProfit_curr'])
    receita_liquida_atual = _get_numeric_value(row, ['receita_liquida_atual', 'totalRevenue', 'Revenue_curr'])
    lucro_bruto_anterior = _get_numeric_value(row, ['lucro_bruto_anterior', 'GrossProfit_prev'])
    receita_liquida_anterior = _get_numeric_value(row, ['receita_liquida_anterior', 'Revenue_prev'])
    debug_valores.update({'lucro_bruto_atual': lucro_bruto_atual, 'receita_liquida_atual': receita_liquida_atual, 
                           'lucro_bruto_anterior': lucro_bruto_anterior, 'receita_liquida_anterior': receita_liquida_anterior})
    
    margem_bruta_atual, margem_bruta_anterior = np.nan, np.nan
    # Prioritize directly provided margin if available
    margem_bruta_atual_direta = _get_numeric_value(row, ['margem_bruta_atual', 'grossMargins', 'GrossMargin_curr'])
    margem_bruta_anterior_direta = _get_numeric_value(row, ['margem_bruta_anterior', 'GrossMargin_prev'])

    if pd.notna(margem_bruta_atual_direta):
        margem_bruta_atual = margem_bruta_atual_direta
    else:
        try:
            if pd.notna(lucro_bruto_atual) and pd.notna(receita_liquida_atual) and receita_liquida_atual != 0:
                margem_bruta_atual = lucro_bruto_atual / receita_liquida_atual
        except Exception: margem_bruta_atual = np.nan
            
    if pd.notna(margem_bruta_anterior_direta):
        margem_bruta_anterior = margem_bruta_anterior_direta
    else:
        try:
            if pd.notna(lucro_bruto_anterior) and pd.notna(receita_liquida_anterior) and receita_liquida_anterior != 0:
                margem_bruta_anterior = lucro_bruto_anterior / receita_liquida_anterior
        except Exception: margem_bruta_anterior = np.nan
    debug_valores.update({'margem_bruta_atual': margem_bruta_atual, 'margem_bruta_anterior': margem_bruta_anterior})
    criterios['aumento_margem_bruta'] = 1 if pd.notna(margem_bruta_atual) and pd.notna(margem_bruta_anterior) and margem_bruta_atual > margem_bruta_anterior else 0

    # 9. Aumento da rotatividade de ativos (DELTA_ASSET_TURN_SCORE)
    rot_ativos_atual, rot_ativos_anterior = np.nan, np.nan
    try:
        if pd.notna(receita_liquida_atual) and pd.notna(ativos_totais_atual) and ativos_totais_atual != 0:
            rot_ativos_atual = receita_liquida_atual / ativos_totais_atual
    except Exception: rot_ativos_atual = np.nan
    try:
        if pd.notna(receita_liquida_anterior) and pd.notna(ativos_totais_anterior) and ativos_totais_anterior != 0:
            rot_ativos_anterior = receita_liquida_anterior / ativos_totais_anterior
    except Exception: rot_ativos_anterior = np.nan
    debug_valores.update({'rot_ativos_atual': rot_ativos_atual, 'rot_ativos_anterior': rot_ativos_anterior})
    criterios['aumento_rotatividade_ativos'] = 1 if pd.notna(rot_ativos_atual) and pd.notna(rot_ativos_anterior) and rot_ativos_atual > rot_ativos_anterior else 0

    score = sum(criterios.values())
    if verbose:
        return score, criterios, debug_valores
    return score

def calcular_altman_z_score(row):
    """
    Calcula o Altman Z-Score (versão para empresas não financeiras).
    Revisado para maior robustez e clareza, conforme solicitado.
    Retorna float ou np.nan.
    Z-Score = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    A = Capital de Giro / Ativos Totais
    B = Lucros Retidos / Ativos Totais
    C = EBIT / Ativos Totais
    D = Valor de Mercado do Patrimônio Líquido / Passivos Totais
    E = Receita Líquida / Ativos Totais
    """
    # Obtenção e conversão robusta dos valores
    ativos_totais = _get_numeric_value(row, ['ativos_totais_atual', 'totalAssets', 'TotalAssets_curr'])
    
    # Capital de Giro (Working Capital) = Ativos Circulantes - Passivos Circulantes
    ativos_circulantes = _get_numeric_value(row, ['ativos_circulantes_atual', 'totalCurrentAssets', 'CurrentAssets_curr'])
    passivos_circulantes = _get_numeric_value(row, ['passivos_circulantes_atual', 'totalCurrentLiabilities', 'CurrentLiab_curr'])
    capital_giro = np.nan
    if pd.notna(ativos_circulantes) and pd.notna(passivos_circulantes):
        capital_giro = ativos_circulantes - passivos_circulantes
    
    # Lucros Retidos (Retained Earnings)
    # Pode ser difícil de obter diretamente. yfinance não fornece facilmente.
    # Fallback: Patrimônio Líquido Atual - (Soma de Capital Social + Reservas de Capital se disponíveis)
    # Ou PL_atual - PL_anterior (se não houve grandes distribuições/recompras que não passaram pelo resultado)
    # Ou, se PL_anterior não disponível, usar PL_atual - LucroLiquido_atual (aproximação grosseira se LL foi todo retido)
    patrimonio_liquido_atual = _get_numeric_value(row, ['patrimonio_liquido_atual', 'totalStockholderEquity', 'StockholdersEquity_curr'])
    patrimonio_liquido_anterior = _get_numeric_value(row, ['patrimonio_liquido_anterior', 'StockholdersEquity_prev'])
    lucro_liquido_atual = _get_numeric_value(row, ['lucro_liquido_atual', 'netIncomeToCommon', 'netIncome'])

    lucro_retido = _get_numeric_value(row, ['lucros_retidos', 'retainedEarnings']) # Tentar campo direto primeiro
    if pd.isna(lucro_retido):
        if pd.notna(patrimonio_liquido_atual) and pd.notna(patrimonio_liquido_anterior):
             # Aproximação: Variação do PL pode indicar lucros retidos + outras variações.
             # Se tivermos lucro líquido do período, PL_anterior + LucroLiquido_atual - Dividendos_atual = PL_atual
             # LucroRetido_periodo = LucroLiquido_atual - Dividendos_atual
             # LucroRetido_acumulado_atual = LucroRetido_acumulado_anterior + LucroRetido_periodo
             # Uma proxy mais simples para Lucros Retidos Acumulados se não disponível:
             # Se PL_atual e PL_anterior estão disponíveis, e assumindo que a variação do PL que não é lucro/prejuízo do período é pequena
             # Esta é uma simplificação, pois PL também é afetado por emissão de ações, etc.
             # A definição original do Z-Score usa o valor acumulado do balanço.
             # Se não tivermos 'retainedEarnings' direto, é um desafio.
             # O mais comum é usar o 'Retained Earnings' do balanço patrimonial.
             # Se yf_ticker_obj.balance_sheet tem 'Retained Earnings', deveria ser pego em obter_dados_fundamentalistas.
             # Por ora, se não vier, vamos deixar como NaN, pois estimativas podem ser muito imprecisas.
             pass # Deixar lucro_retido como NaN se não encontrado diretamente ou via fallback robusto.

    # EBIT (Lucro Antes de Juros e Impostos)
    # yfinance financials: 'EBIT', 'EarningsBeforeInterestAndTaxes'
    ebit = _get_numeric_value(row, ['ebit', 'EBIT', 'earningsBeforeInterestAndTaxes'])
    if pd.isna(ebit):
        ebit_margin = _get_numeric_value(row, ['ebit_margin', 'ebitdaMargins']) # ebitdaMargin as proxy if ebitMargin not there
        receita_liquida = _get_numeric_value(row, ['receita_liquida_atual', 'totalRevenue'])
        if pd.notna(ebit_margin) and pd.notna(receita_liquida):
            ebit = ebit_margin * receita_liquida
            
    # Valor de Mercado do Patrimônio Líquido (Market Capitalization)
    valor_mercado_pl = _get_numeric_value(row, ['marketCap', 'valor_mercado'])
    if pd.isna(valor_mercado_pl) and pd.notna(patrimonio_liquido_atual):
         # Fallback para book value of equity se market cap não disponível (menos ideal)
         # Não recomendado para Z-score original, mas como último recurso.
         # valor_mercado_pl = patrimonio_liquido_atual
         pass # Melhor deixar NaN se marketCap não estiver disponível.

    # Passivos Totais (Total Liabilities)
    # totalDebt do yfinance.info é Dívida Total. Precisamos de Passivos Totais.
    # Passivos Totais = Ativos Totais - Patrimônio Líquido Total
    total_passivo = _get_numeric_value(row, ['total_passivo', 'totalLiab', 'TotalLiabilities_curr'])
    if pd.isna(total_passivo):
        if pd.notna(ativos_totais) and pd.notna(patrimonio_liquido_atual):
            total_passivo = ativos_totais - patrimonio_liquido_atual

    # Receita Líquida (Net Sales/Revenue)
    receita_liquida = _get_numeric_value(row, ['receita_liquida_atual', 'totalRevenue', 'Revenue_curr'])

    # Verificar se todos os componentes essenciais para o cálculo de A, B, C, D, E estão presentes
    if pd.isna(ativos_totais) or ativos_totais == 0: # Denominador comum
        return np.nan

    # Cálculo dos componentes do Z-Score
    A, B, C, D, E = 0, 0, 0, 0, 0 # Default to 0 if component cannot be calculated

    if pd.notna(capital_giro):
        A = capital_giro / ativos_totais
    
    # Lucro Retido é crucial e muitas vezes não vem fácil. Se NaN, B será 0.
    if pd.notna(lucro_retido):
        B = lucro_retido / ativos_totais
    
    if pd.notna(ebit):
        C = ebit / ativos_totais
    
    if pd.notna(valor_mercado_pl) and pd.notna(total_passivo) and total_passivo != 0:
        D = valor_mercado_pl / total_passivo
    elif pd.notna(valor_mercado_pl) and pd.notna(total_passivo) and total_passivo == 0 and valor_mercado_pl > 0 : # No liabilities, solvent
        D = 100 # Assign a large number as a proxy for very low risk from leverage
    elif pd.notna(valor_mercado_pl) and pd.notna(total_passivo) and total_passivo == 0 and valor_mercado_pl == 0 : # Edge case
        D = 1 # Neutral if both are zero
        
    if pd.notna(receita_liquida):
        E = receita_liquida / ativos_totais

    # Fórmula do Z-Score
    try:
        z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
        return z_score
    except Exception:
        return np.nan

def calcular_beneish_m_score(row):
    """
    Calcula o Beneish M-Score para detectar manipulação contábil.
    Revisado para maior robustez e uso de proxies onde possível.
    Retorna float ou np.nan.
    M-Score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI - 0.327*LVGI + 4.679*TATA
    (Nota: fórmula original tem +4.679*TATA, mas algumas fontes usam -0.327*LVGI + 4.679*TATA, outras + LVGI - TATA. Verificarei a mais comum.)
    A fórmula original de Beneish (1999) é: M = -4.84 + 0.920(DSRI) + 0.528(GMI) + 0.404(AQI) + 0.892(SGI) + 0.115(DEPI) – 0.172(SGAI) + 4.679(TATA) – 0.327(LVGI)
    O script original tinha +LVGI e -TATA, o que é diferente. Usarei a fórmula de 1999.
    """
    # --- Obtenção dos dados para os índices --- 
    # DSRI = (Contas a Receber_t / Receita_t) / (Contas a Receber_t-1 / Receita_t-1)
    contas_receber_t = _get_numeric_value(row, ['contas_a_receber_atual', 'netReceivables', 'NetReceivables_curr'])
    receita_t = _get_numeric_value(row, ['receita_liquida_atual', 'totalRevenue', 'Revenue_curr'])
    contas_receber_t_1 = _get_numeric_value(row, ['contas_a_receber_anterior', 'NetReceivables_prev'])
    receita_t_1 = _get_numeric_value(row, ['receita_liquida_anterior', 'Revenue_prev'])

    dsri = 1.0 # Default se não calculável
    try:
        if pd.notna(contas_receber_t) and pd.notna(receita_t) and receita_t != 0 and \
           pd.notna(contas_receber_t_1) and pd.notna(receita_t_1) and receita_t_1 != 0:
            dsri_t = contas_receber_t / receita_t
            dsri_t_1 = contas_receber_t_1 / receita_t_1
            if dsri_t_1 != 0: dsri = dsri_t / dsri_t_1
    except Exception: dsri = 1.0

    # GMI = (Margem Bruta_t-1 / Margem Bruta_t)
    # Margem Bruta = (Receita - CMV) / Receita = Lucro Bruto / Receita
    margem_bruta_t = _get_numeric_value(row, ['margem_bruta_atual', 'grossMargins'])
    margem_bruta_t_1 = _get_numeric_value(row, ['margem_bruta_anterior', 'GrossMargin_prev'])
    if pd.isna(margem_bruta_t):
        lucro_bruto_t = _get_numeric_value(row, ['lucro_bruto_atual', 'grossProfit'])
        if pd.notna(lucro_bruto_t) and pd.notna(receita_t) and receita_t != 0: margem_bruta_t = lucro_bruto_t / receita_t
    if pd.isna(margem_bruta_t_1):
        lucro_bruto_t_1 = _get_numeric_value(row, ['lucro_bruto_anterior', 'GrossProfit_prev'])
        if pd.notna(lucro_bruto_t_1) and pd.notna(receita_t_1) and receita_t_1 != 0: margem_bruta_t_1 = lucro_bruto_t_1 / receita_t_1

    gmi = 1.0
    try:
        if pd.notna(margem_bruta_t_1) and pd.notna(margem_bruta_t) and margem_bruta_t != 0:
            gmi = margem_bruta_t_1 / margem_bruta_t
    except Exception: gmi = 1.0

    # AQI = [1 - (Ativos Circ. Op._t + Imobilizado_t) / Ativos Totais_t] / [1 - (Ativos Circ. Op._t-1 + Imobilizado_t-1) / Ativos Totais_t-1]
    # Ativos Circ. Op. = Ativos Circ. - Caixa e Equivalentes
    # Simplificação: (1 - (PPE_t + CurrentAssets_t) / TotalAssets_t) / (1 - (PPE_t-1 + CurrentAssets_t-1) / TotalAssets_t-1)
    # PPE = Property, Plant, Equipment (Imobilizado)
    caixa_equivalentes_t = _get_numeric_value(row, ['caixa_e_equivalentes_atual', 'cash', 'Cash_curr'])
    ativos_circulantes_t = _get_numeric_value(row, ['ativos_circulantes_atual', 'totalCurrentAssets'])
    imobilizado_t = _get_numeric_value(row, ['imobilizado_atual', 'propertyPlantEquipment', 'PPE_curr'])
    ativos_totais_t = _get_numeric_value(row, ['ativos_totais_atual', 'totalAssets'])
    
    caixa_equivalentes_t_1 = _get_numeric_value(row, ['caixa_e_equivalentes_anterior', 'Cash_prev'])
    ativos_circulantes_t_1 = _get_numeric_value(row, ['ativos_circulantes_anterior', 'CurrentAssets_prev'])
    imobilizado_t_1 = _get_numeric_value(row, ['imobilizado_anterior', 'PPE_prev'])
    ativos_totais_t_1 = _get_numeric_value(row, ['ativos_totais_anterior', 'TotalAssets_prev'])

    aqi = 1.0
    try:
        # Ativos não produtores de caixa (Non-cash generating current assets)
        ncga_t = 0; ncga_t_1 = 0
        if pd.notna(ativos_circulantes_t) and pd.notna(caixa_equivalentes_t): ncga_t = ativos_circulantes_t - caixa_equivalentes_t
        elif pd.notna(ativos_circulantes_t): ncga_t = ativos_circulantes_t # Se caixa não disponível, usa AC total
        
        if pd.notna(ativos_circulantes_t_1) and pd.notna(caixa_equivalentes_t_1): ncga_t_1 = ativos_circulantes_t_1 - caixa_equivalentes_t_1
        elif pd.notna(ativos_circulantes_t_1): ncga_t_1 = ativos_circulantes_t_1

        # Ativos de qualidade (produtivos)
        asset_quality_t_num = 0; asset_quality_t_1_num = 0
        if pd.notna(ncga_t) and pd.notna(imobilizado_t): asset_quality_t_num = ncga_t + imobilizado_t
        elif pd.notna(ativos_circulantes_t) and pd.notna(imobilizado_t): asset_quality_t_num = ativos_circulantes_t + imobilizado_t # Fallback
        
        if pd.notna(ncga_t_1) and pd.notna(imobilizado_t_1): asset_quality_t_1_num = ncga_t_1 + imobilizado_t_1
        elif pd.notna(ativos_circulantes_t_1) and pd.notna(imobilizado_t_1): asset_quality_t_1_num = ativos_circulantes_t_1 + imobilizado_t_1 # Fallback

        if pd.notna(asset_quality_t_num) and pd.notna(ativos_totais_t) and ativos_totais_t != 0 and \
           pd.notna(asset_quality_t_1_num) and pd.notna(ativos_totais_t_1) and ativos_totais_t_1 != 0:
            aq_ratio_t = (ativos_totais_t - asset_quality_t_num) / ativos_totais_t # Fração de ativos que SÃO caixa ou não são CA nem PPE
            aq_ratio_t_1 = (ativos_totais_t_1 - asset_quality_t_1_num) / ativos_totais_t_1
            if aq_ratio_t_1 != 0: aqi = aq_ratio_t / aq_ratio_t_1
    except Exception: aqi = 1.0

    # SGI = Receita_t / Receita_t-1
    sgi = 1.0
    try:
        if pd.notna(receita_t) and pd.notna(receita_t_1) and receita_t_1 != 0:
            sgi = receita_t / receita_t_1
    except Exception: sgi = 1.0

    # DEPI = (Depreciação_t-1 / (Depreciação_t-1 + Imobilizado_t-1)) / (Depreciação_t / (Depreciação_t + Imobilizado_t))
    depreciacao_t = _get_numeric_value(row, ['depreciacao_amortizacao_atual', 'depreciation', 'DepreciationAmortization_curr'])
    depreciacao_t_1 = _get_numeric_value(row, ['depreciacao_amortizacao_anterior', 'DepreciationAmortization_prev'])
    # Imobilizado (PPE) já obtido para AQI

    depi = 1.0
    try:
        if pd.notna(depreciacao_t_1) and pd.notna(imobilizado_t_1) and (depreciacao_t_1 + imobilizado_t_1) != 0 and \
           pd.notna(depreciacao_t) and pd.notna(imobilizado_t) and (depreciacao_t + imobilizado_t) != 0:
            rate_t_1 = depreciacao_t_1 / (depreciacao_t_1 + imobilizado_t_1)
            rate_t = depreciacao_t / (depreciacao_t + imobilizado_t)
            if rate_t != 0: depi = rate_t_1 / rate_t
    except Exception: depi = 1.0

    # SGAI = (Despesas SGA_t / Receita_t) / (Despesas SGA_t-1 / Receita_t-1)
    # SGA = Selling, General & Administrative Expenses
    sga_t = _get_numeric_value(row, ['despesas_sga_atual', 'sellingGeneralAdministrative', 'SGAExpense_curr'])
    sga_t_1 = _get_numeric_value(row, ['despesas_sga_anterior', 'SGAExpense_prev'])

    sgai = 1.0
    try:
        if pd.notna(sga_t) and pd.notna(receita_t) and receita_t != 0 and \
           pd.notna(sga_t_1) and pd.notna(receita_t_1) and receita_t_1 != 0:
            sgai_ratio_t = sga_t / receita_t
            sgai_ratio_t_1 = sga_t_1 / receita_t_1
            if sgai_ratio_t_1 != 0: sgai = sgai_ratio_t / sgai_ratio_t_1
    except Exception: sgai = 1.0

    # LVGI = (Passivo Total_t / Ativo Total_t) / (Passivo Total_t-1 / Ativo Total_t-1)
    # Passivo Total = Dívida Total (Curto + Longo Prazo)
    # Ou Ativos Totais - Patrimônio Líquido
    total_passivo_t = _get_numeric_value(row, ['total_passivo_atual', 'totalLiab'])
    if pd.isna(total_passivo_t) and pd.notna(ativos_totais_t) and pd.notna(_get_numeric_value(row, ['patrimonio_liquido_atual', 'totalStockholderEquity'])):
        total_passivo_t = ativos_totais_t - _get_numeric_value(row, ['patrimonio_liquido_atual', 'totalStockholderEquity'])
        
    total_passivo_t_1 = _get_numeric_value(row, ['total_passivo_anterior', 'TotalLiab_prev'])
    if pd.isna(total_passivo_t_1) and pd.notna(ativos_totais_t_1) and pd.notna(_get_numeric_value(row, ['patrimonio_liquido_anterior', 'StockholdersEquity_prev'])):
        total_passivo_t_1 = ativos_totais_t_1 - _get_numeric_value(row, ['patrimonio_liquido_anterior', 'StockholdersEquity_prev'])

    lvgi = 1.0
    try:
        if pd.notna(total_passivo_t) and pd.notna(ativos_totais_t) and ativos_totais_t != 0 and \
           pd.notna(total_passivo_t_1) and pd.notna(ativos_totais_t_1) and ativos_totais_t_1 != 0:
            lvg_ratio_t = total_passivo_t / ativos_totais_t
            lvg_ratio_t_1 = total_passivo_t_1 / ativos_totais_t_1
            if lvg_ratio_t_1 != 0: lvgi = lvg_ratio_t / lvg_ratio_t_1
    except Exception: lvgi = 1.0

    # TATA = (Lucro Líquido_t - Fluxo de Caixa Operacional_t) / Ativos Totais_t
    # Lucro Líquido e CFO já obtidos para Piotroski
    lucro_liquido_t = _get_numeric_value(row, ['lucro_liquido_atual', 'netIncomeToCommon', 'netIncome'])
    cfo_t = _get_numeric_value(row, ['cfo_atual', 'operatingCashflow', 'totalCashFromOperatingActivities'])
    # Ativos Totais (ativos_totais_t) já obtido

    tata = 0.0 # Default
    try:
        if pd.notna(lucro_liquido_t) and pd.notna(cfo_t) and pd.notna(ativos_totais_t) and ativos_totais_t != 0:
            tata = (lucro_liquido_t - cfo_t) / ativos_totais_t
    except Exception: tata = 0.0

    # Beneish M-Score Formula (Beneish 1999)
    # M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI – 0.172*SGAI + 4.679*TATA – 0.327*LVGI
    try:
        m_score = (
            -4.84 
            + (0.920 * dsri) 
            + (0.528 * gmi) 
            + (0.404 * aqi) 
            + (0.892 * sgi) 
            + (0.115 * depi) 
            - (0.172 * sgai) 
            + (4.679 * tata) 
            - (0.327 * lvgi)
        )
        return m_score
    except Exception:
        return np.nan

# --- Funções de Análise de Portfólio (Exemplos do script original, não focos da revisão atual) ---
def calcular_score_setorial(row, media_setor: dict, campos_multiplo: list):
    score = 0
    count_valid_metrics = 0
    for campo in campos_multiplo:
        valor = _get_numeric_value(row, [campo])
        media = pd.to_numeric(media_setor.get(campo), errors='coerce')
        if pd.notna(valor) and pd.notna(media) and media != 0:
            score += (valor / media)
            count_valid_metrics +=1
    return score / count_valid_metrics if count_valid_metrics > 0 else np.nan
    
def calcular_value_composite_score(df_fundamental, metrics_config):
    if df_fundamental.empty:
        return pd.Series(dtype=float)
    ranked_df = pd.DataFrame(index=df_fundamental.index)
    for metric, ranking_type in metrics_config.items():
        if metric in df_fundamental.columns:
            numeric_metric = pd.to_numeric(df_fundamental[metric], errors='coerce')
            if numeric_metric.isna().all(): # Skip if all values are NaN for this metric
                print(f"Warning: Metric {metric} is all NaN, skipping for Value Composite Score.")
                continue
            if ranking_type == 'lower_is_better':
                ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=True) 
            elif ranking_type == 'higher_is_better':
                # Original Piotroski is higher is better, so rank descending for pct=True
                ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=False)
            else: # Default to lower is better if unspecified
                 ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=True)
        else:
            print(f"Warning: Metric {metric} not found in fundamental data for Value Composite Score.")
    rank_cols = [col for col in ranked_df.columns if '_rank' in col]
    if not rank_cols:
        return pd.Series(dtype=float, index=df_fundamental.index)
    composite_score = ranked_df[rank_cols].mean(axis=1)
    return composite_score

def calcular_volatilidade_garch(returns_series, p=1, q=1):
    if not isinstance(returns_series, pd.Series) or returns_series.empty or returns_series.isna().all():
        print(f"Input for GARCH is not a valid Series or is empty/all NaN: {getattr(returns_series, 'name', 'Unnamed Series')}")
        return np.nan
    if len(returns_series.dropna()) < (p + q + 20): # Ensure enough non-NaN data points
        print(f"Not enough non-NaN data points for GARCH model on series: {returns_series.name}")
        return np.nan
    try:
        scaled_returns = returns_series.dropna() * 100
        if scaled_returns.empty:
             print(f"Scaled returns are empty after dropna for GARCH: {returns_series.name}")
             return np.nan
        model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1)
        forecasted_std_dev_daily_scaled = np.sqrt(forecast.variance.iloc[-1,0])
        forecasted_std_dev_daily = forecasted_std_dev_daily_scaled / 100.0
        return forecasted_std_dev_daily * np.sqrt(252) # Annualized
    except Exception as e:
        # Catch specific errors if possible, e.g., from arch_model
        print(f"Error fitting GARCH for {returns_series.name}: {e}")
        return np.nan

def get_fama_french_factors(start_date, end_date, risk_free_rate_series=None):
    print("Fetching Fama-French factor proxies...")
    factor_tickers = {
        'MKT_PROXY': '^GSPC', # S&P 500 as market proxy
        'SMB_SMALL': '^RUT',  # Russell 2000 for small cap
        'SMB_LARGE': '^GSPC', # Using S&P 500 again for large cap proxy (common simplification)
        'HML_VALUE': 'IVE',   # iShares S&P 500 Value ETF
        'HML_GROWTH':'IVW',  # iShares S&P 500 Growth ETF
        'WML_MOM': 'MTUM'    # iShares MSCI USA Momentum Factor ETF
    }
    
    downloaded_data = yf.download(list(factor_tickers.values()), start=start_date, end=end_date, progress=False)
    if downloaded_data.empty:
        print("Could not download factor proxy data (empty main DataFrame).")
        return pd.DataFrame()

    # Determine if 'Adj Close' or 'Close' is available and use it consistently
    price_col_type = None
    if isinstance(downloaded_data.columns, pd.MultiIndex):
        if 'Adj Close' in downloaded_data.columns.levels[0]:
            price_col_type = 'Adj Close'
        elif 'Close' in downloaded_data.columns.levels[0]:
            price_col_type = 'Close'
    else: # Single level columns
        if 'Adj Close' in downloaded_data.columns:
            price_col_type = 'Adj Close'
        elif 'Close' in downloaded_data.columns:
            price_col_type = 'Close'
            
    if price_col_type is None:
        print(f"Error: Neither 'Adj Close' nor 'Close' found for factor data. Available: {downloaded_data.columns}")
        return pd.DataFrame()

    factor_data_raw = downloaded_data[price_col_type]
    if factor_data_raw.empty:
        print(f"Extracted factor price data ('{price_col_type}') is empty.")
        return pd.DataFrame()

    factor_returns = factor_data_raw.pct_change().dropna(how='all')
    factors_df = pd.DataFrame(index=factor_returns.index)

    # Risk-Free Rate
    if risk_free_rate_series is None:
        rf_data_yf = yf.download('^IRX', start=start_date, end=end_date, progress=False) # 13 Week Treasury Bill
        if not rf_data_yf.empty:
            rf_price_col = 'Adj Close' if 'Adj Close' in rf_data_yf.columns else 'Close'
            if rf_price_col in rf_data_yf.columns:
                rf_series_raw = rf_data_yf[rf_price_col]
                if pd.api.types.is_numeric_dtype(rf_series_raw):
                    # ^IRX is an annualized yield percentage. Convert to daily decimal.
                    daily_rf = (rf_series_raw / 100) / 252 
                    factors_df['RF'] = daily_rf.reindex(factors_df.index, method='ffill')
                else:
                    print("Warning: ^IRX data for risk-free rate is not numeric. Using default.")
                    factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
            else:
                print("Warning: Could not find 'Adj Close' or 'Close' for ^IRX. Using default.")
                factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
        else:
            print("Warning: Could not fetch ^IRX for risk-free rate. Using default.")
            factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
    else:
        factors_df['RF'] = risk_free_rate_series.reindex(factors_df.index, method='ffill')

    factors_df['RF'].fillna(method='ffill', inplace=True)
    factors_df['RF'].fillna(method='bfill', inplace=True) # Fill any remaining NaNs at the beginning
    factors_df.dropna(subset=['RF'], inplace=True) # Drop if RF still NaN (e.g. all inputs were NaN)
    
    if factors_df.empty and not factor_returns.empty : # If RF failed but returns exist, reindex for common dates
        factors_df = pd.DataFrame(index=factor_returns.index)
        factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)

    # Align factor_returns to factors_df index (which is now based on RF availability)
    factor_returns = factor_returns.reindex(factors_df.index).dropna(how='all')
    factors_df = factors_df.reindex(factor_returns.index).dropna(how='all')

    # Mkt-RF
    if factor_tickers['MKT_PROXY'] in factor_returns.columns and 'RF' in factors_df.columns:
        factors_df['Mkt-RF'] = factor_returns[factor_tickers['MKT_PROXY']] - factors_df['RF']
    
    # SMB (Small Minus Big)
    if factor_tickers['SMB_SMALL'] in factor_returns.columns and factor_tickers['SMB_LARGE'] in factor_returns.columns:
        factors_df['SMB'] = factor_returns[factor_tickers['SMB_SMALL']] - factor_returns[factor_tickers['SMB_LARGE']]

    # HML (High Minus Low)
    if factor_tickers['HML_VALUE'] in factor_returns.columns and factor_tickers['HML_GROWTH'] in factor_returns.columns:
        factors_df['HML'] = factor_returns[factor_tickers['HML_VALUE']] - factor_returns[factor_tickers['HML_GROWTH']]

    # WML (Winners Minus Losers) or MOM (Momentum)
    if factor_tickers['WML_MOM'] in factor_returns.columns and 'RF' in factors_df.columns:
        # Some definitions of WML are just the momentum factor returns, others are Mom - RF.
        # Here, using Mom returns directly as WML, or (Mom - RF) if that's the convention desired.
        # For Fama-French 5-factor, it's often RMW (Robust Minus Weak profitability) and CMA (Conservative Minus Aggressive investment)
        # The original script used 'WML', implying a momentum factor. MTUM is a momentum ETF.
        factors_df['WML'] = factor_returns[factor_tickers['WML_MOM']] # Using raw momentum factor returns
        # If WML should be Mom - RF: factors_df['WML'] = factor_returns[factor_tickers['WML_MOM']] - factors_df['RF']

    final_factors = ['Mkt-RF', 'SMB', 'HML', 'WML', 'RF']
    factors_df = factors_df[[col for col in final_factors if col in factors_df.columns]].copy()
    factors_df.dropna(inplace=True)
    return factors_df

def ajustar_retornos_esperados(base_retornos_medios_anuais, df_fundamental, alphas, arima_forecasts, quant_value_scores, piotroski_scores):
    adjusted_retornos = base_retornos_medios_anuais.copy()
    for ativo in adjusted_retornos.index:
        adjustment_factor = 1.0
        if piotroski_scores is not None and ativo in piotroski_scores and pd.notna(piotroski_scores[ativo]):
            pscore = piotroski_scores[ativo]
            if pscore <= 2: adjustment_factor *= 0.9
            elif pscore >=7: adjustment_factor *= 1.1
        if quant_value_scores is not None and ativo in quant_value_scores and pd.notna(quant_value_scores[ativo]):
            qscore = quant_value_scores[ativo]
            adjustment_factor *= (1 + (qscore - 0.5) * 0.2)
        adjusted_retornos[ativo] *= adjustment_factor
        if alphas is not None and ativo in alphas and pd.notna(alphas[ativo]):
            adjusted_retornos[ativo] += alphas[ativo]
        if arima_forecasts is not None and ativo in arima_forecasts and pd.notna(arima_forecasts[ativo]):
            annualized_arima_forecast = arima_forecasts[ativo] * 252
            adjusted_retornos[ativo] = (adjusted_retornos[ativo] * 0.8) + (annualized_arima_forecast * 0.2)
    return adjusted_retornos

def estimar_fatores_alpha_beta(asset_returns_series, factor_df):
    if not isinstance(asset_returns_series, pd.Series) or asset_returns_series.empty or asset_returns_series.isna().all():
        return np.nan, {}
    if factor_df.empty or not all(col in factor_df.columns for col in ['Mkt-RF', 'RF']): # Basic check
        print(f"Factor DataFrame is empty or missing essential columns for {asset_returns_series.name}")
        return np.nan, {}

    # Ensure asset_returns_series is daily if factors are daily
    # (Assuming both are already in compatible frequency)
    y_excess = asset_returns_series - factor_df['RF']
    
    # Align data by index (date)
    common_index = y_excess.index.intersection(factor_df.index)
    y_aligned = y_excess.loc[common_index].dropna()
    
    # Select factor columns for regression (excluding RF as it's already used for excess returns)
    factor_cols_for_reg = [col for col in ['Mkt-RF', 'SMB', 'HML', 'WML'] if col in factor_df.columns]
    if not factor_cols_for_reg:
        print(f"No valid factor columns found for regression for {asset_returns_series.name}")
        return np.nan, {}
        
    X_aligned = factor_df.loc[common_index, factor_cols_for_reg].dropna(how='any', axis=0)
    
    # Final alignment after individual NaNs dropped
    final_common_index = y_aligned.index.intersection(X_aligned.index)
    y_final = y_aligned.loc[final_common_index]
    X_final = X_aligned.loc[final_common_index]

    if len(y_final) < max(20, len(X_final.columns) + 2): # Min observations for regression
        print(f"Not enough common/valid data points for factor regression on {asset_returns_series.name} (have {len(y_final)})")
        return np.nan, {}

    X_final_with_const = sm.add_constant(X_final)
    
    try:
        model = sm.OLS(y_final, X_final_with_const, missing='drop').fit()
        alpha_daily = model.params.get('const', np.nan)
        alpha_annualized = alpha_daily * 252 if pd.notna(alpha_daily) else np.nan
        betas = model.params.drop('const', errors='ignore').to_dict()
        return alpha_annualized, betas
    except Exception as e:
        asset_name_for_log = asset_returns_series.name if isinstance(asset_returns_series, pd.Series) else "unknown_asset"
        print(f"Error in OLS regression for {asset_name_for_log}: {e}")
        return np.nan, {}

def prever_retornos_arima(returns_series, order=(5,1,0)):
    if not isinstance(returns_series, pd.Series) or returns_series.empty or returns_series.isna().all():
        print(f"Input for ARIMA is not a valid Series or is empty/all NaN: {getattr(returns_series, 'name', 'Unnamed Series')}")
        return np.nan
    
    cleaned_returns = returns_series.dropna()
    if len(cleaned_returns) < (sum(order) + 20): # Heuristic for minimum data length
        print(f"Not enough non-NaN data points for ARIMA model on series: {returns_series.name} (have {len(cleaned_returns)})")
        return np.nan
    try:
        model = ARIMA(cleaned_returns, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast.iloc[0] if not forecast.empty else np.nan
    except Exception as e:
        print(f"Error fitting ARIMA for {returns_series.name}: {e}")
        return np.nan

# --- Main Execution / Example Usage (Illustrative) ---
if __name__ == '__main__':
    # Exemplo de uso:
    ativos_exemplo = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'WEGE3.SA', 'MGLU3.SA', 'VIIA3.SA'] # Adicione um ticker com poucos dados como VIIA3.SA
    
    print("--- Obtendo Dados Fundamentalistas Detalhados ---")
    df_fundamentos = obter_dados_fundamentalistas_detalhados_br(ativos_exemplo)
    print(df_fundamentos.head())
    print("\nCampos disponíveis:", df_fundamentos.columns.tolist())

def calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco):
    retorno_portfolio = np.sum(retornos_medios_anuais * pesos)
    volatilidade_portfolio = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia_anual, pesos)))
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / volatilidade_portfolio if volatilidade_portfolio != 0 else -np.inf
    return retorno_portfolio, volatilidade_portfolio, sharpe_ratio

    # Aplicar cálculos de scores
def otimizar_portfolio_scipy(
    ativos,
    df_retornos_historicos,
    df_fundamental_completo=None,
    fama_french_factors=None,
    taxa_livre_risco=RISK_FREE_RATE_DEFAULT,
    pesos_atuais=None,
    restricoes_pesos_min_max=None,
    objetivo='max_sharpe'
):
    if df_retornos_historicos.empty or len(ativos) == 0:
        return None, None, []
    retornos_considerados = df_retornos_historicos[ativos].copy()
    if retornos_considerados.shape[1] != len(ativos) or retornos_considerados.isnull().values.any():
        print("Warning: Missing return data for some assets or all assets. Dropping NaNs.")
        retornos_considerados.dropna(axis=1, how='all', inplace=True)
        retornos_considerados.dropna(axis=0, how='any', inplace=True)
        ativos = list(retornos_considerados.columns)
        if not ativos:
            print("Error: No valid asset data left after cleaning for optimization.")
            return None, None, []
    retornos_medios_diarios = retornos_considerados.mean()
    base_retornos_medios_anuais = retornos_medios_diarios * 252
    matriz_covariancia_diaria = retornos_considerados.cov()
    matriz_covariancia_anual_historica = matriz_covariancia_diaria * 252
    num_ativos = len(ativos)
    adj_retornos_esperados = base_retornos_medios_anuais.copy()
    adj_matriz_covariancia = matriz_covariancia_anual_historica.copy()
    alphas = {}
    arima_forecasts = {}
    garch_volatilities = {}
    if df_fundamental_completo is not None and not df_fundamental_completo.empty:
        if 'Piotroski_F_Score' not in df_fundamental_completo.columns:
            df_fundamental_completo['Piotroski_F_Score'] = df_fundamental_completo.apply(calcular_piotroski_f_score_br, axis=1)
        vc_metrics = {
            'trailingPE': 'lower_is_better', 
            'priceToBook': 'lower_is_better', 
            'enterpriseToEbitda': 'lower_is_better',
            'dividendYield': 'higher_is_better', 
            'returnOnEquity': 'higher_is_better',
            'netMargin': 'higher_is_better'
        }
        if 'Quant_Value_Score' not in df_fundamental_completo.columns:
            df_fundamental_completo['Quant_Value_Score'] = calcular_value_composite_score(df_fundamental_completo, vc_metrics)
        piotroski_scores_series = df_fundamental_completo['Piotroski_F_Score'].reindex(ativos)
        quant_value_scores_series = df_fundamental_completo['Quant_Value_Score'].reindex(ativos)
        for ativo in ativos:
            asset_returns = retornos_considerados[ativo].dropna()
            if fama_french_factors is not None and not fama_french_factors.empty:
                alpha, _ = estimar_fatores_alpha_beta(asset_returns, fama_french_factors)
                alphas[ativo] = alpha
            arima_forecasts[ativo] = prever_retornos_arima(asset_returns)
            garch_vol = calcular_volatilidade_garch(asset_returns)
            if pd.notna(garch_vol) and garch_vol > 0:
                if ativo in adj_matriz_covariancia.index:
                    adj_matriz_covariancia.loc[ativo, ativo] = garch_vol**2
                    garch_volatilities[ativo] = garch_vol
        adj_retornos_esperados = ajustar_retornos_esperados(
            base_retornos_medios_anuais, 
            df_fundamental_completo.reindex(ativos),
            alphas, 
            arima_forecasts, 
            quant_value_scores_series,
            piotroski_scores_series
        )
    else:
        print("No fundamental data or factors provided for advanced optimization. Using historical mean/covariance.")
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    def portfolio_return(weights, expected_returns):
        return np.sum(expected_returns * weights)
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        p_return = portfolio_return(weights, expected_returns)
        p_volatility = portfolio_volatility(weights, cov_matrix)
        if p_volatility == 0: return -np.inf
        return -(p_return - risk_free_rate) / p_volatility
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = []
    default_min_b, default_max_b = 0.0, 1.0
    if isinstance(restricoes_pesos_min_max, tuple) and len(restricoes_pesos_min_max) == 2:
        default_min_b, default_max_b = restricoes_pesos_min_max[0], restricoes_pesos_min_max[1]
    for ativo in ativos:
        min_b, max_b = default_min_b, default_max_b
        if isinstance(restricoes_pesos_min_max, dict) and ativo in restricoes_pesos_min_max:
            min_b, max_b = restricoes_pesos_min_max[ativo]
        elif pesos_atuais and ativo in pesos_atuais:
            pass
        bounds.append((min_b, max_b))
    initial_weights = np.array([1/num_ativos] * num_ativos)
    if pesos_atuais and len(pesos_atuais) == num_ativos:
        current_weights_array = np.array([pesos_atuais.get(a, 0) for a in ativos])
        if np.isclose(np.sum(current_weights_array), 1.0):
            initial_weights = current_weights_array
        else:
            print("Warning: Sum of provided current weights is not 1. Using equal initial weights.")
    if objetivo == 'max_sharpe':
        opt_args = (adj_retornos_esperados.values, adj_matriz_covariancia.values, taxa_livre_risco)
        opt_func = neg_sharpe_ratio
    elif objetivo == 'min_volatility':
        opt_args = (adj_matriz_covariancia.values,)
        opt_func = lambda w, cov: portfolio_volatility(w, cov)
    else:
        raise ValueError("Invalid objective function specified.")
    optimized_result = minimize(opt_func, initial_weights, args=opt_args,
                                method='SLSQP', bounds=bounds, constraints=constraints)
    if not optimized_result.success:
        print(f"Optimization failed: {optimized_result.message}")
    optimal_weights = optimized_result.x
    optimal_weights[optimal_weights < 1e-4] = 0
    optimal_weights = optimal_weights / np.sum(optimal_weights)
    ret_opt, vol_opt, sharpe_opt = calcular_metricas_portfolio(optimal_weights, adj_retornos_esperados, adj_matriz_covariancia, taxa_livre_risco)
    portfolio_otimizado = {
        'pesos': dict(zip(ativos, optimal_weights)),
        'retorno_esperado': ret_opt,
        'volatilidade': vol_opt,
        'sharpe_ratio': sharpe_opt,
        'garch_volatilities': garch_volatilities if garch_volatilities else None,
        'alphas': alphas if alphas else None,
        'arima_forecasts': arima_forecasts if arima_forecasts else None
    }
    fronteira_pontos_simulados = []
    return portfolio_otimizado, fronteira_pontos_simulados

def otimizar_portfolio_markowitz_mc(ativos, df_retornos_historicos, taxa_livre_risco=RISK_FREE_RATE_DEFAULT, num_portfolios_simulados=100000):
    if df_retornos_historicos.empty or len(ativos) == 0:
        return None, None, []
    retornos_considerados = df_retornos_historicos[ativos].copy()
    retornos_considerados.dropna(axis=1, how='all', inplace=True)
    retornos_considerados.dropna(axis=0, how='any', inplace=True)
    ativos = list(retornos_considerados.columns)
    if not ativos:
        return None, None, []
    retornos_medios_diarios = retornos_considerados.mean()
    matriz_covariancia_diaria = retornos_considerados.cov()
    num_ativos_calc = len(ativos)
    retornos_medios_anuais = retornos_medios_diarios * 252
    matriz_covariancia_anual = matriz_covariancia_diaria * 252
    resultados_lista = []
    for _ in range(num_portfolios_simulados):
        pesos = np.random.random(num_ativos_calc)
        pesos /= np.sum(pesos)
        retorno, volatilidade, sharpe = calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco)
        resultados_lista.append({'retorno': retorno, 'volatilidade': volatilidade, 'sharpe': sharpe, 'pesos': dict(zip(ativos, pesos))})
    if not resultados_lista:
        return None, None, []
    portfolio_max_sharpe_dict = max(resultados_lista, key=lambda x: x['sharpe'])
    portfolio_max_sharpe = {
        'pesos': portfolio_max_sharpe_dict['pesos'],
        'retorno_esperado': portfolio_max_sharpe_dict['retorno'],
        'volatilidade': portfolio_max_sharpe_dict['volatilidade'],
        'sharpe_ratio': portfolio_max_sharpe_dict['sharpe']
    }
    return portfolio_max_sharpe, resultados_lista

def sugerir_alocacao_novo_aporte(
    current_portfolio_composition_values: dict, 
    new_capital: float,
    target_portfolio_weights_decimal: dict,
):
    if new_capital <= 1e-6: return {}, 0.0
    current_portfolio_value = sum(current_portfolio_composition_values.values())
    final_portfolio_value = current_portfolio_value + new_capital
    purchases_to_reach_target = {}
    all_involved_assets = set(current_portfolio_composition_values.keys()) | set(target_portfolio_weights_decimal.keys())
    for asset_ticker in all_involved_assets:
        current_asset_value = current_portfolio_composition_values.get(asset_ticker, 0.0)
        target_asset_final_value = target_portfolio_weights_decimal.get(asset_ticker, 0.0) * final_portfolio_value
        amount_to_buy = max(0, target_asset_final_value - current_asset_value)
        if amount_to_buy > 1e-6: purchases_to_reach_target[asset_ticker] = amount_to_buy
    total_capital_needed_for_target = sum(purchases_to_reach_target.values())
    actual_purchases = {}
    surplus_capital = 0.0
    if total_capital_needed_for_target <= 1e-6:
        relevant_target_assets = {t: w for t, w in target_portfolio_weights_decimal.items() if w > 1e-6}
        sum_target_weights_for_distribution = sum(relevant_target_assets.values())
        if sum_target_weights_for_distribution > 1e-6:
            for asset_ticker, weight in relevant_target_assets.items():
                actual_purchases[asset_ticker] = (weight / sum_target_weights_for_distribution) * new_capital
        else: surplus_capital = new_capital
    elif total_capital_needed_for_target <= new_capital:
        actual_purchases = purchases_to_reach_target.copy()
        surplus_capital_after_reaching_target = new_capital - total_capital_needed_for_target
        if surplus_capital_after_reaching_target > 1e-6:
            relevant_target_assets = {t: w for t, w in target_portfolio_weights_decimal.items() if w > 1e-6}
            sum_target_weights_for_distribution = sum(relevant_target_assets.values())
            if sum_target_weights_for_distribution > 1e-6:
                for asset_ticker, weight in relevant_target_assets.items():
                    additional_buy = (weight / sum_target_weights_for_distribution) * surplus_capital_after_reaching_target
                    actual_purchases[asset_ticker] = actual_purchases.get(asset_ticker, 0) + additional_buy
                surplus_capital = 0.0
            else: surplus_capital = surplus_capital_after_reaching_target
        else: surplus_capital = 0.0
    else:
        for asset_ticker, needed_amount in purchases_to_reach_target.items():
            actual_purchases[asset_ticker] = (needed_amount / total_capital_needed_for_target) * new_capital
        surplus_capital = 0.0
    actual_purchases = {k: v for k, v in actual_purchases.items() if v > 0.01}
    return actual_purchases, surplus_capital

# --- Main execution block for testing ---
if __name__ == '__main__':
    print("Running financial_analyzer_enhanced.py tests...")
    test_ativos = ['AAPL', 'MSFT', 'GOOGL']
    start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    print("\n--- Testing Historical Data ---")
    df_retornos = obter_dados_historicos_yf(test_ativos, start_date_str=start_date, end_date_str=end_date)
    if not df_retornos.empty:
        print(f"Successfully fetched historical returns for {len(df_retornos.columns)} assets. Shape: {df_retornos.shape}")
        print(df_retornos.head())
    else:
        print("Failed to fetch historical returns.")
    print("\n--- Testing Fundamental Data ---")
    df_fund_data = obter_dados_fundamentalistas_detalhados_br(test_ativos)
    if not df_fund_data.empty:
        df_fund_data['Piotroski_F_Score'], df_fund_data['Piotroski_F_Detalhes'] = zip(*df_fund_data.apply(lambda row: calcular_piotroski_f_score_br(row, verbose=True), axis=1))
        df_fund_data['Altman_Z_Score'] = df_fund_data.apply(calcular_altman_z_score, axis=1)
        df_fund_data['Beneish_M_Score'] = df_fund_data.apply(calcular_beneish_m_score, axis=1)
        # E assim por diante...
        vc_metrics_test = {
            'trailingPE': 'lower_is_better', 'priceToBook': 'lower_is_better', 
            'enterpriseToEbitda': 'lower_is_better', 'dividendYield': 'higher_is_better',
            'returnOnEquity': 'higher_is_better', 'netMargin': 'higher_is_better'
        }
        df_fund_data['Quant_Value_Score'] = calcular_value_composite_score(df_fund_data, vc_metrics_test)
        print(df_fund_data[['ticker', 'Piotroski_F_Score', 'Quant_Value_Score']].head())
    else:
        print("Failed to fetch fundamental data.")
    print("\n--- Testing Fama-French Factors ---")
    ff_factors = get_fama_french_factors(start_date, end_date)
    if not ff_factors.empty:
        print(f"Successfully fetched Fama-French factor proxies. Shape: {ff_factors.shape}")
        print(ff_factors.head())
    else:
        print("Failed to fetch Fama-French factors.")
    print("\n--- Testing Advanced Portfolio Optimization (SciPy) ---")
    if not df_retornos.empty and not df_fund_data.empty and not ff_factors.empty:
        df_fund_data.set_index('ticker', inplace=True, drop=False)
        optimized_portfolio_advanced, _ = otimizar_portfolio_scipy(
            ativos=test_ativos,
            df_retornos_historicos=df_retornos,
            df_fundamental_completo=df_fund_data,
            fama_french_factors=ff_factors,
            taxa_livre_risco=RISK_FREE_RATE_DEFAULT,
            objetivo='max_sharpe'
        )
        if optimized_portfolio_advanced:
            print("Advanced Optimized Portfolio (Max Sharpe):")
            for k, v in optimized_portfolio_advanced.items():
                if k == 'pesos': print(f"  {k}: { {tk: f'{p*100:.2f}%' for tk,p in v.items()} }")
                elif isinstance(v, dict): print(f"  {k}: Present")
                elif isinstance(v, float): print(f"  {k}: {v:.4f}")
                else: print(f"  {k}: {v}")
        else:
            print("Advanced portfolio optimization failed.")
    else:
        print("Skipping advanced optimization test due to missing data (returns, fundamentals, or factors).")
    print("\n--- Testing Original Markowitz Optimization (Monte Carlo) ---")
    if not df_retornos.empty:
        portfolio_sharpe_mc, _ = otimizar_portfolio_markowitz_mc(test_ativos, df_retornos, taxa_livre_risco=RISK_FREE_RATE_DEFAULT)
        if portfolio_sharpe_mc:
            print("MC Optimized Portfolio (Max Sharpe):")
            print(f"  Pesos: { {tk: f'{p*100:.2f}%' for tk,p in portfolio_sharpe_mc['pesos'].items()} }")
            print(f"  Retorno Esperado: {portfolio_sharpe_mc['retorno_esperado']:.4f}")
            print(f"  Volatilidade: {portfolio_sharpe_mc['volatilidade']:.4f}")
            print(f"  Sharpe Ratio: {portfolio_sharpe_mc['sharpe_ratio']:.4f}")
        else:
            print("MC portfolio optimization failed.")
    else:
        print("Skipping MC optimization test due to missing return data.")
    print("\nfinancial_analyzer_enhanced.py tests completed.")
# Compatibilidade: garantir nome antigo para import do Streamlit
calcular_piotroski_f_score = calcular_piotroski_f_score_br
beneish_m_score = calcular_beneish_m_score
obter_dados_fundamentalistas_detalhados = obter_dados_fundamentalistas_detalhados_br


