import os
import sys
import logging
import requests
from datetime import datetime, timedelta
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
        logging.info("텔레그램 메시지 전송 성공")
        return True
    except Exception as e:
        logging.error(f"Telegram 전송 실패: {str(e)}")
        return False

def fetch_nasdaq_data(query_date):
    """NASDAQ 배당 데이터 조회"""
    url = f'https://api.nasdaq.com/api/calendar/dividends?date={query_date}'
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
        logging.info(f"NASDAQ API 조회 성공: {query_date}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"NASDAQ API 요청 실패 ({query_date}): {str(e)}")
        return None

def calculate_last_purchase_date(ex_date):
    """배당을 받기 위한 최종 매수일 계산 (배당락일 전 영업일)"""
    try:
        ex_date_obj = pd.to_datetime(ex_date)
        # 배당락일 전 영업일 (실제로는 전일까지 매수해야 함)
        last_purchase_date = ex_date_obj - pd.offsets.BusinessDay(1)
        return last_purchase_date.strftime('%Y-%m-%d')
    except Exception as e:
        logging.error(f"최종 매수일 계산 오류: {str(e)}")
        return None

def is_purchase_deadline_valid(ex_date, current_date, min_days_ahead=1):
    """매수 기한이 유효한지 확인 (최소 N일 이상의 여유가 있는지)"""
    try:
        ex_date_obj = pd.to_datetime(ex_date).date()
        # 배당락일이 현재일보다 min_days_ahead일 이상 미래에 있어야 함
        min_valid_date = current_date + timedelta(days=min_days_ahead)
        return ex_date_obj >= min_valid_date
    except Exception as e:
        logging.error(f"매수 기한 검증 오류: {str(e)}")
        return False

def process_stock(stock, current_date):
    """개별 종목 처리"""
    try:
        # 필수 필드 검증
        required_fields = ['symbol', 'companyName', 'dividend_Ex_Date',
                          'dividend_Rate', 'indicated_Annual_Dividend', 'payment_Date']
        if not all(stock.get(field) for field in required_fields):
            logging.warning(f"필수 데이터 누락: {stock.get('symbol', 'Unknown')}")
            return None

        # 배당락일 파싱 및 검증
        ex_date_str = stock['dividend_Ex_Date']
        
        # 매수 기한이 유효한지 확인 (최소 1일 여유)
        if not is_purchase_deadline_valid(ex_date_str, current_date, min_days_ahead=1):
            logging.info(f"제외: {stock['symbol']} (배당락일 {ex_date_str} - 매수 기한 부족)")
            return None

        # 주가 정보 조회
        ticker = stock['symbol']
        try:
            ticker_info = yf.Ticker(ticker)
            hist = ticker_info.history(period='5d')
            
            if hist.empty:
                logging.warning(f"주가 데이터 없음: {ticker}")
                return None
                
            # 가장 최근 유효한 주가 사용
            price = hist.iloc[-1].Close
            if pd.isna(price):
                valid_prices = hist.Close.dropna()
                if valid_prices.empty:
                    logging.warning(f"유효한 주가 데이터 없음: {ticker}")
                    return None
                price = valid_prices.iloc[-1]
                
        except Exception as e:
            logging.warning(f"주가 조회 실패 {ticker}: {str(e)}")
            return None

        # 배당 정보 파싱
        try:
            dividend_rate = float(stock['dividend_Rate'])
            annual_dividend = float(stock['indicated_Annual_Dividend'])
        except (ValueError, TypeError) as e:
            logging.warning(f"배당 정보 파싱 실패 {ticker}: {str(e)}")
            return None

        # 배당 수익률 계산
        dividend_yield = (annual_dividend / price) * 100 if price > 0 else 0

        # 고배당 종목만 필터링 (3% 이상)
        if dividend_yield >= 3:
            last_purchase_date = calculate_last_purchase_date(ex_date_str)
            
            return {
                'Symbol': ticker,
                'Name': stock['companyName'],
                'Ex-Date': ex_date_str,
                'Last_Purchase_Date': last_purchase_date,
                'Dividend': dividend_rate,
                'Annual Dividend': annual_dividend,
                'Dividend Yield': f"{dividend_yield:.2f}%",
                'Dividend_Yield_Value': dividend_yield,
                'Payment Date': stock['payment_Date'],
                'Current Price': f"${price:.2f}"
            }
        else:
            logging.info(f"제외: {ticker} (배당수익률 {dividend_yield:.2f}% < 3%)")
            
    except Exception as e:
        logging.error(f"종목 처리 오류 {stock.get('symbol', 'Unknown')}: {str(e)}")
    
    return None

def fetch_multiple_days_data(start_date, days_ahead=14):
    """여러 날짜의 배당 데이터를 조회하여 통합"""
    all_stocks = []
    
    for i in range(days_ahead):
        query_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        logging.info(f"배당 데이터 조회 중: {query_date}")
        
        data = fetch_nasdaq_data(query_date)
        if data and data.get('data'):
            rows = data['data'].get('calendar', {}).get('rows', [])
            all_stocks.extend(rows)
            logging.info(f"{query_date}: {len(rows)}개 종목 발견")
    
    logging.info(f"총 {len(all_stocks)}개 종목 데이터 수집 완료")
    return all_stocks

def main():
    try:
        # 현재 동부시간 기준
        now_et = datetime.now(EASTERN_TZ)
        current_date = now_et.date()
        current_time_str = now_et.strftime('%Y-%m-%d %H:%M')
        
        logging.info(f"프로그램 시작 - 동부시간: {current_time_str}")

        # 향후 2주간의 배당 데이터 조회
        all_dividend_stocks = fetch_multiple_days_data(now_et, days_ahead=14)
        
        if not all_dividend_stocks:
            logging.warning("배당 데이터를 가져오지 못했습니다.")
            return

        logging.info(f"총 {len(all_dividend_stocks)}개 종목 처리 시작")

        # 종목 처리 (멀티스레딩)
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda s: process_stock(s, current_date), all_dividend_stocks))
            filtered_stocks = [res for res in results if res is not None]

        # 중복 제거 (동일 심볼)
        unique_stocks = {}
        for stock in filtered_stocks:
            symbol = stock['Symbol']
            if symbol not in unique_stocks:
                unique_stocks[symbol] = stock
            else:
                # 더 높은 배당수익률을 가진 것으로 유지
                if stock['Dividend_Yield_Value'] > unique_stocks[symbol]['Dividend_Yield_Value']:
                    unique_stocks[symbol] = stock

        final_stocks = list(unique_stocks.values())
        
        # 배당 수익률 기준 내림차순 정렬
        final_stocks.sort(key=lambda x: x['Dividend_Yield_Value'], reverse=True)

        logging.info(f"최종 {len(final_stocks)}개 고배당 종목 선별 완료")

        # 메시지 생성 및 전송
        if not final_stocks:
            message = (
                f"<b>[{current_time_str} ET] 미국주식 고배당 종목 알림</b>\n\n"
                f"현재 조건에 맞는 고배당 종목이 없습니다.\n"
                f"(배당수익률 3% 이상, 매수 여유시간 1일 이상)"
            )
            send_telegram(message)
            return

        # 메시지를 여러 개로 분할하여 전송 (텔레그램 메시지 길이 제한 고려)
        chunk_size = 8
        total_stocks = len(final_stocks)
        total_parts = (total_stocks + chunk_size - 1) // chunk_size

        for part in range(total_parts):
            start = part * chunk_size
            end = min(start + chunk_size, total_stocks)
            current_chunk = final_stocks[start:end]
            part_num = part + 1
            
            message = (
                f"<b>[고배당 종목 알림] Part {part_num}/{total_parts}</b>\n"
                f"📅 동부시간: {current_time_str}\n"
                f"📊 총 {total_stocks}개 종목 (배당수익률 3% 이상)\n"
                f"⚠️ 배당 받으려면 배당락일 전 영업일까지 매수 필요\n\n"
            )
            
            for idx, stock in enumerate(current_chunk, 1):
                global_idx = start + idx
                
                # 최종 매수일까지 남은 일수 계산
                try:
                    last_purchase_date = pd.to_datetime(stock['Last_Purchase_Date']).date()
                    days_left = (last_purchase_date - current_date).days
                    urgency_emoji = "🔥" if days_left <= 2 else "⏰" if days_left <= 5 else "📅"
                except:
                    days_left = "?"
                    urgency_emoji = "📅"
                
                message += (
                    f"{urgency_emoji} <b>{global_idx}. {stock['Symbol']}</b>\n"
                    f"🏢 {stock['Name']}\n"
                    f"💰 현재가: {stock['Current Price']}\n"
                    f"📈 배당수익률: <b>{stock['Dividend Yield']}</b>\n"
                    f"💵 배당금: ${stock['Dividend']} (연 ${stock['Annual Dividend']})\n"
                    f"🗓️ 배당락일: {stock['Ex-Date']}\n"
                    f"⏰ 최종매수일: {stock['Last_Purchase_Date']} ({days_left}일 남음)\n"
                    f"💳 지급일: {stock['Payment Date']}\n\n"
                )

            if not send_telegram(message):
                logging.error(f"Part {part_num} 전송 실패")
                sys.exit(1)

        logging.info("모든 메시지 전송 완료")

    except Exception as e:
        error_msg = f"프로그램 실행 오류: {str(e)}"
        logging.critical(error_msg, exc_info=True)
        
        # 에러 알림도 텔레그램으로 전송
        send_telegram(f"🚨 <b>배당 알림 프로그램 오류</b>\n\n{error_msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
