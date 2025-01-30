import os
import sys
import logging
import requests
from datetime import datetime
import pytz
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import Retry, HTTPAdapter

# 환경 변수 검증
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram 환경 변수가 설정되지 않았습니다.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dividend_alert.log'),
        logging.StreamHandler()
    ]
)

EASTERN_TZ = pytz.timezone('US/Eastern')

def send_telegram(message):
    """텔레그램 메시지 전송 함수"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Telegram 전송 실패: {str(e)}")
        return False

def fetch_nasdaq_data(target_date):
    """NASDAQ 배당 데이터 조회 (2영업일 후 기준)"""
    url = f'https://api.nasdaq.com/api/calendar/dividends?date={target_date}'
    headers = {
        'accept': 'application/json, text/plain, */*',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36',
        'referer': 'https://www.nasdaq.com/'
    }

    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"NASDAQ API 요청 실패: {str(e)}")
        return None

def calculate_target_date():
    """2영업일 후 날짜 계산"""
    try:
        now_et = datetime.now(EASTERN_TZ)
        target_date = now_et + pd.offsets.BusinessDay(2)
        return target_date.strftime('%Y-%m-%d')
    except Exception as e:
        logging.error(f"목표일 계산 오류: {str(e)}")
        return None

def process_stock(stock, current_date):
    """개별 종목 처리"""
    try:
        # 필수 필드 검증
        required_fields = ['symbol', 'companyName', 'dividend_Ex_Date',
                          'dividend_Rate', 'indicated_Annual_Dividend', 'payment_Date']
        if not all(stock.get(field) for field in required_fields):
            logging.warning(f"필수 데이터 누락: {stock.get('symbol')}")
            return None

        # 배당락일 파싱
        ex_date_str = stock['dividend_Ex_Date']
        ex_date = pd.to_datetime(ex_date_str).date()
        
        # 현재 날짜와 비교 (2영업일 후 기준)
        if ex_date <= current_date:
            logging.info(f"제외: {stock['symbol']} (배당락일 {ex_date_str} 기준 구매기한 종료)")
            return None

        # 주가 정보 조회
        ticker = stock['symbol']
        ticker_info = yf.Ticker(ticker)
        hist = ticker_info.history(period='5d')
        
        if hist.empty:
            logging.warning(f"주가 데이터 없음: {ticker}")
            return None
            
        # 주가 처리
        price = hist.iloc[-1].Close
        if pd.isna(price):
            price = hist.Close.dropna().iloc[-1]

        # 배당 수익률 계산
        dividend_rate = float(stock['dividend_Rate'])
        annual_dividend = float(stock['indicated_Annual_Dividend'])
        dividend_yield = (annual_dividend / price) * 100 if price else 0

        if dividend_yield >= 3:
            return {
                'Symbol': ticker,
                'Name': stock['companyName'],
                'Ex-Date': ex_date.strftime('%Y-%m-%d'),
                'Dividend': dividend_rate,
                'Annual Dividend': annual_dividend,
                'Dividend Yield': f"{dividend_yield:.2f}%",
                'Dividend_Yield_Value': dividend_yield,
                'Payment Date': stock['payment_Date'],
                'Current Price': f"${price:.2f}"
            }
            
    except Exception as e:
        logging.error(f"종목 처리 오류 {stock.get('symbol')}: {str(e)}")
    return None

def main():
    try:
        # 현재 동부시간 기준
        now_et = datetime.now(EASTERN_TZ)
        current_date = now_et.date()
        current_time_str = now_et.strftime('%Y-%m-%d %H:%M')
        
        # 2영업일 후 날짜 계산
        target_date = calculate_target_date()
        if not target_date:
            raise ValueError("목표일 계산 실패")

        # NASDAQ 데이터 조회
        data = fetch_nasdaq_data(target_date)
        if not data or not data.get('data'):
            logging.warning("배당 데이터를 가져오지 못했습니다.")
            return

        # 종목 처리
        rows = data['data'].get('calendar', {}).get('rows', [])
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(lambda s: process_stock(s, current_date), rows)
            filtered_stocks = [res for res in results if res]

        # 배당 수익률 기준 정렬
        filtered_stocks.sort(key=lambda x: x['Dividend_Yield_Value'], reverse=True)

        # 메시지 생성 및 전송
        if not filtered_stocks:
            logging.info("전송할 종목이 없습니다.")
            return

        chunk_size = 10
        total_stocks = len(filtered_stocks)
        total_parts = (total_stocks + chunk_size - 1) // chunk_size

        for part in range(total_parts):
            start = part * chunk_size
            end = start + chunk_size
            current_chunk = filtered_stocks[start:end]
            part_num = part + 1
            
            message = (
                f"<b>[{target_date}] 미국주식 고배당 종목 알림 ({total_stocks}건)</b>\n"
                f"※ 동부시간 기준 {current_time_str}\n"
                f"※ 최종 매수 기한: 배당락일 당일 00:00 ET 기준\n\n"
            )
            
            for idx, stock in enumerate(current_chunk, 1):
                global_idx = start + idx
                cutoff_date = (pd.to_datetime(stock['Ex-Date']) - pd.offsets.BusinessDay(2)).strftime('%Y-%m-%d')
                message += (
                    f"<b>{global_idx}. {stock['Symbol']}</b> ({stock['Name']})\n"
                    f"├ 배당락일: {stock['Ex-Date']}\n"
                    f"├ 최종 매수일: {cutoff_date}\n"
                    f"├ 현재 가격: {stock['Current Price']}\n"
                    f"├ 배당 수익률: {stock['Dividend Yield']}\n"
                    f"├ 배당금: ${stock['Dividend']} (연간 ${stock['Annual Dividend']})\n"
                    f"└ 지급일: {stock['Payment Date']}\n\n"
                )

            if not send_telegram(message):
                sys.exit(1)

    except Exception as e:
        logging.critical(f"프로그램 실행 오류: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
