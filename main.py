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

# 시간대 설정
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

def fetch_nasdaq_data(date):
    """NASDAQ 배당 데이터 조회"""
    url = f'https://api.nasdaq.com/api/calendar/dividends?date={date}'
    headers = {
        'accept': 'application/json, text/plain, */*',
        'user-agent': 'Mozilla/5.0',
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

def parse_date(date_str):
    """날짜 문자열 파싱 함수"""
    for fmt in ('%Y-%m-%d', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    logging.error(f"날짜 형식 오류: {date_str}")
    return None

def calculate_cutoff_date(ex_date):
    """배당락일 기준 2영업일 전 날짜 계산"""
    try:
        us_bd = pd.offsets.CustomBusinessDay(calendar=pd.tseries.holiday.USFederalHolidayCalendar())
        cutoff_date = ex_date - us_bd * 2
        return cutoff_date
    except Exception as e:
        logging.error(f"날짜 계산 오류: {str(e)}")
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
        ex_date = parse_date(ex_date_str)
        if not ex_date:
            return None

        cutoff_date = calculate_cutoff_date(ex_date)
        if not cutoff_date or current_date > cutoff_date.date():
            logging.info(f"제외: {stock['symbol']} (배당락일 {ex_date_str} 기준 구매기한 종료)")
            return None

        # 주가 정보 조회
        ticker = stock['symbol']
        ticker_info = yf.Ticker(ticker)
        hist = ticker_info.history(period='5d')

        if hist.empty:
            logging.warning(f"주가 데이터 없음: {ticker}")
            return None

        # 주가 추출
        price = hist.iloc[-1].Close
        if pd.isna(price):
            price = hist.Close.dropna().iloc[-1]

        # 배당 수익률 계산
        dividend_rate = float(stock['dividend_Rate'])
        annual_dividend = float(stock['indicated_Annual_Dividend'])
        if price > 0:
            dividend_yield = (annual_dividend / price) * 100
        else:
            dividend_yield = 0

        if dividend_yield >= 3:
            return {
                'Symbol': ticker,
                'Name': stock['companyName'],
                'Ex-Date': ex_date.strftime('%Y-%m-%d'),
                'Cutoff-Date': cutoff_date.strftime('%Y-%m-%d'),
                'Dividend': dividend_rate,
                'Annual Dividend': annual_dividend,
                'Dividend Yield': f"{dividend_yield:.2f}%",
                'Dividend_Yield_Value': dividend_yield,
                'Payment Date': stock['payment_Date'],
                'Current Price': f"${price:.2f}",
                # 'Market Cap': f"${market_cap/1e9:.2f}B"  # 시가총액 정보 제거
            }
    except Exception as e:
        logging.error(f"종목 처리 오류 {stock.get('symbol')}: {str(e)}")
    return None

def main():
    try:
        # 현재 동부시간 기준 날짜
        now_et = datetime.now(EASTERN_TZ)
        today_et = now_et.strftime('%Y-%m-%d')
        current_date = now_et.date()

        # NASDAQ 데이터 조회
        data = fetch_nasdaq_data(today_et)
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
                f"<b>[{today_et}] 미국주식 고배당 종목 알림 ({total_stocks}건)</b>\n"
                f"※ 동부시간 기준 {now_et.strftime('%Y-%m-%d %H:%M')}\n"
                f"※ 최종 매수 기한: 배당락일 2영업일 전까지\n\n"
            )

            for idx, stock in enumerate(current_chunk, 1):
                global_idx = start + idx
                message += (
                    f"<b>{global_idx}. {stock['Symbol']}</b> ({stock['Name']})\n"
                    f"├ 배당락일: {stock['Ex-Date']}\n"
                    f"├ 최종 매수일: {stock['Cutoff-Date']}\n"
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
