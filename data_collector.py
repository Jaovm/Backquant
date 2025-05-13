#!/usr/bin/python3.11
import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import json
import pandas as pd
import os
from datetime import datetime

def fetch_and_save_data():
    client = ApiClient()
    # Adicionando .SA para ativos brasileiros e incluindo o índice Bovespa
    tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'MGLU3.SA', 'WEGE3.SA', '^BVSP']
    data_dir = "/home/ubuntu/collected_data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Iniciando coleta de dados para: {', '.join(tickers)}")

    for ticker in tickers:
        print(f"Processando {ticker}...")

        # 1. Coletar dados históricos (get_stock_chart)
        try:
            print(f"  Coletando dados de gráfico para {ticker}...")
            chart_data = client.call_api('YahooFinance/get_stock_chart',
                                         query={'symbol': ticker,
                                                'range': '10y',
                                                'interval': '1d',
                                                'includeAdjustedClose': True})
            
            if chart_data and chart_data.get('chart') and chart_data['chart'].get('result'):
                result = chart_data['chart']['result'][0]
                timestamps = result.get('timestamp', [])
                if not timestamps:
                    print(f"    Dados de timestamp não encontrados para {ticker}.")
                    continue

                dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
                
                ohlcv = result.get('indicators', {}).get('quote', [{}])[0]
                adjclose_list = result.get('indicators', {}).get('adjclose', [{}])[0].get('adjclose', [])

                df_data = {
                    'Date': dates,
                    'Open': ohlcv.get('open', [None]*len(dates)),
                    'High': ohlcv.get('high', [None]*len(dates)),
                    'Low': ohlcv.get('low', [None]*len(dates)),
                    'Close': ohlcv.get('close', [None]*len(dates)),
                    'Volume': ohlcv.get('volume', [None]*len(dates)),
                    'Adj Close': adjclose_list if adjclose_list else [None]*len(dates)
                }
                
                df = pd.DataFrame(df_data)
                # Remover linhas onde todos os valores de preço/volume são None (pode acontecer com alguns tickers/intervalos)
                price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                df.dropna(subset=price_cols, how='all', inplace=True)
                
                if not df.empty:
                    # Sanitize ticker name for filename
                    safe_ticker_name = ticker.replace('^', 'INDEX_')
                    file_path = os.path.join(data_dir, f"{safe_ticker_name}_chart_data.csv")
                    df.to_csv(file_path, index=False)
                    print(f"    Dados de gráfico para {ticker} salvos em {file_path}")
                else:
                    print(f"    DataFrame vazio para dados de gráfico de {ticker} após limpeza.")

            else:
                print(f"    Não foi possível obter dados de gráfico para {ticker}. Resposta: {chart_data.get('error') if chart_data else 'Nenhuma resposta'}")
        except Exception as e:
            print(f"    Erro ao coletar dados de gráfico para {ticker}: {e}")

        # 2. Coletar insights (get_stock_insights) - Não aplicável para índices como ^BVSP
        if not ticker.startswith('^'): # Índices geralmente não têm 'insights' da mesma forma
            try:
                print(f"  Coletando insights para {ticker}...")
                insights_data = client.call_api('YahooFinance/get_stock_insights', query={'symbol': ticker})
                
                if insights_data and insights_data.get('finance') and insights_data['finance'].get('result'):
                    # Sanitize ticker name for filename
                    safe_ticker_name = ticker.replace('^', 'INDEX_')
                    file_path = os.path.join(data_dir, f"{safe_ticker_name}_insights_data.json")
                    with open(file_path, 'w') as f:
                        json.dump(insights_data['finance']['result'], f, indent=4)
                    print(f"    Insights para {ticker} salvos em {file_path}")
                else:
                    print(f"    Não foi possível obter insights para {ticker}. Resposta: {insights_data.get('error') if insights_data else 'Nenhuma resposta'}")
            except Exception as e:
                print(f"    Erro ao coletar insights para {ticker}: {e}")
        else:
            print(f"    Coleta de insights pulada para o índice {ticker}.")

    print("Coleta de dados concluída.")

if __name__ == "__main__":
    fetch_and_save_data()

