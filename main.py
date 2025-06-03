import os
import sys
import logging
import requests
from datetime import datetime
import pytz
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay # CustomBusinessDay 추가
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import Retry, HTTPAdapter

# 환경 변수 검증
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    # 로깅 설정이 이전에 되어야 raise 이전에 로그를 남길 수 있습니다.
    # 하지만, 여기서는 스크립트 실행의 필수 조건이므로, 일단 그대로 둡니다.
    # 실제 운영 환경에서는 설정 로드와 로깅 초기화를 더 정교하게 관리할 수 있습니다.
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

from datetime import date # Add date import

# 설정값 (환경 변수 또는 기본값)
DEFAULT_DIVIDEND_YIELD_THRESHOLD = 3.0
try:
    DIVIDEND_YIELD_THRESHOLD = float(os.environ.get('DIVIDEND_YIELD_THRESHOLD', DEFAULT_DIVIDEND_YIELD_THRESHOLD))
except ValueError:
    logging.warning(f"환경 변수 DIVIDEND_YIELD_THRESHOLD가 유효한 숫자가 아닙니다. 기본값 {DEFAULT_DIVIDEND_YIELD_THRESHOLD}을 사용합니다.")
    DIVIDEND_YIELD_THRESHOLD = DEFAULT_DIVIDEND_YIELD_THRESHOLD

DEFAULT_NASDAQ_API_URL = 'https://api.nasdaq.com/api/calendar/dividends'
NASDAQ_API_URL = os.environ.get('NASDAQ_API_URL', DEFAULT_NASDAQ_API_URL)

logging.info(f"사용 중인 배당 수익률 임계값: {DIVIDEND_YIELD_THRESHOLD}{' (기본값)' if DIVIDEND_YIELD_THRESHOLD == DEFAULT_DIVIDEND_YIELD_THRESHOLD and not os.environ.get('DIVIDEND_YIELD_THRESHOLD') else ''}")
logging.info(f"사용 중인 NASDAQ API URL: {NASDAQ_API_URL}{' (기본값)' if NASDAQ_API_URL == DEFAULT_NASDAQ_API_URL and not os.environ.get('NASDAQ_API_URL') else ''}")


EASTERN_TZ = pytz.timezone('US/Eastern')

# Holiday Calendar Setup
# Start with federal holidays for the current year and next to be safe for date calculations
current_year_for_holidays = datetime.now(EASTERN_TZ).year
# Get holidays for a window around current year to ensure calculations are robust
# Using date objects for holiday list
federal_holidays_list = USFederalHolidayCalendar().holidays(
    start=date(current_year_for_holidays - 1, 1, 1),
    end=date(current_year_for_holidays + 2, 12, 31)
).date # Convert to array of date objects

# Add specific known stock market holidays not covered by USFederalHolidayCalendar
# Good Friday dates need to be calculated or sourced. For 2024: March 29.
# For demonstration, adding 2024 and 2025. A more dynamic source would be better for production.
additional_stock_market_holidays = [
    date(2024, 3, 29),  # Good Friday 2024
    date(2025, 4, 18),  # Good Friday 2025
]

# Combine holiday lists and remove duplicates
# Ensure all are date objects for consistency
APP_HOLIDAYS = list(set(list(federal_holidays_list) + additional_stock_market_holidays))
APP_HOLIDAYS.sort()
logging.info(f"{len(APP_HOLIDAYS)}일의 공휴일이 로드되었습니다 (연방 공휴일 + 추가된 주식 시장 공휴일).")


def send_telegram(message):
# EASTERN_TZ, send_telegram 등 나머지 코드는 여기에 이어집니다.
# 중복된 로깅 설정과 EASTERN_TZ 정의를 제거합니다.
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
        logging.info(f"Telegram 메시지 성공적으로 전송 (Chat ID: {TELEGRAM_CHAT_ID}).")
        return True
    except Exception as e:
        logging.error(f"Telegram 전송 실패 (Chat ID: {TELEGRAM_CHAT_ID}): {str(e)}")
        return False

def fetch_nasdaq_data(target_date):
    """NASDAQ 배당 데이터 조회 (2영업일 후 기준)"""
    # NASDAQ_API_URL 변수를 사용하도록 수정
    url = f'{NASDAQ_API_URL}?date={target_date}'
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
        logging.info(f"NASDAQ API로부터 데이터 성공적으로 수신 (URL: {url}).")
        return response.json()
    except requests.exceptions.ConnectionError as e:
        logging.error(f"NASDAQ API 연결 오류 (URL: {url}): {str(e)}")
        return None
    except requests.exceptions.HTTPError as e:
        logging.error(f"NASDAQ API HTTP 오류 (URL: {url}): {str(e)}")
        return None
    except requests.exceptions.Timeout as e:
        logging.error(f"NASDAQ API 요청 시간 초과 (URL: {url}): {str(e)}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"NASDAQ API 요청 실패 (URL: {url}): {str(e)}")
        return None

def calculate_target_date():
    """2영업일 후 날짜 계산 (미국 공휴일 반영)"""
    try:
        now_et = datetime.now(EASTERN_TZ)
        # APP_HOLIDAYS 리스트를 사용하여 영업일 계산
        target_date = now_et + CustomBusinessDay(2, holidays=APP_HOLIDAYS)
        return target_date.strftime('%Y-%m-%d')
    except Exception as e:
        logging.error(f"목표일 계산 오류: {str(e)}")
        return None

# --- process_stock 리팩토링 시작 ---

def _validate_stock_data(stock_data, required_fields):
    """주식 데이터의 필수 필드가 모두 있는지 검증합니다.

    Args:
        stock_data (dict): 검증할 주식 데이터.
        required_fields (list): 필수 필드명의 리스트.

    Returns:
        bool: 모든 필수 필드가 존재하고 비어있지 않으면 True, 그렇지 않으면 False.
    """
    if not stock_data:
        return False
    for field in required_fields:
        if not stock_data.get(field): # .get()은 필드가 없거나 값이 None일 경우 False로 평가될 수 있는 값을 반환
            return False
    return True

def _get_stock_price_info(ticker_symbol):
    """Yahoo Finance를 통해 특정 티커의 최신 종가를 조회합니다.

    Args:
        ticker_symbol (str): 조회할 주식 티커 심볼.

    Returns:
        float or None: 최신 종가를 float 형태로 반환하거나, 조회 실패 시 None을 반환합니다.
    """
    try:
        ticker_info = yf.Ticker(ticker_symbol)
        hist = ticker_info.history(period='5d') # 최근 5일 데이터 조회

        if hist.empty:
            logging.warning(f"주가 데이터 없음 (hist empty): {ticker_symbol}")
            return None
            
        # 최신 유효 종가 탐색
        price = hist.iloc[-1].Close
        if pd.isna(price): # 최신일 종가가 NaN인 경우
            # NaN이 아닌 마지막 종가 사용
            valid_prices = hist.Close.dropna()
            if valid_prices.empty:
                logging.warning(f"유효한 종가 데이터 없음 (all NaN): {ticker_symbol}")
                return None
            price = valid_prices.iloc[-1]

        return float(price)
    except Exception as e:
        # yfinance 호출 관련 예외 등
        logging.error(f"주가 정보 조회 중 오류 발생 ({ticker_symbol}): {str(e)}")
        return None

def process_stock(stock_data, current_date): # stock -> stock_data, current_date는 현재 미사용이나 유지
    """NASDAQ API로부터 받은 개별 종목 데이터를 처리하여 배당 정보를 추출합니다.

    Args:
        stock_data (dict): NASDAQ API로부터 받은 개별 종목 정보.
        current_date (datetime.date): 현재 날짜 (ET 기준). (현재 로직에서는 직접 사용되지 않음)

    Returns:
        dict or None: 필터링 및 처리된 종목 정보를 담은 딕셔너리, 또는 조건 미달/오류 시 None.
    """
    required_fields = ['symbol', 'companyName', 'dividend_Ex_Date',
                       'dividend_Rate', 'indicated_Annual_Dividend', 'payment_Date']

    if not _validate_stock_data(stock_data, required_fields):
        logging.warning(f"필수 데이터 누락 또는 빈 값: {stock_data.get('symbol', 'N/A')}")
        return None

    ticker = stock_data['symbol']

    # 배당락일 파싱 (문자열 -> date 객체)
    try:
        ex_date_str = stock_data['dividend_Ex_Date']
        ex_date = pd.to_datetime(ex_date_str).date()
    except ValueError as e:
        logging.warning(f"배당락일 형식 오류 ({ticker}): {ex_date_str} - {str(e)}")
        return None

    # 주가 정보 조회
    price = _get_stock_price_info(ticker)
    if price is None: # _get_stock_price_info 내부에서 이미 로깅됨
        return None

    try:
        # 배당 수익률 계산
        # dividend_Rate는 단일 배당금, indicated_Annual_Dividend는 연간 총 예상 배당금
        annual_dividend = float(stock_data['indicated_Annual_Dividend'])
        dividend_rate = float(stock_data['dividend_Rate']) # 개별 배당금도 저장해둠

        if price == 0: # 0으로 나누기 방지
            dividend_yield = 0.0
        else:
            dividend_yield = (annual_dividend / price) * 100

        # DIVIDEND_YIELD_THRESHOLD (환경 변수 또는 기본값) 기준으로 필터링
        if dividend_yield >= DIVIDEND_YIELD_THRESHOLD:
            return {
                'Symbol': ticker,
                'Name': stock_data['companyName'],
                'Ex-Date': ex_date.strftime('%Y-%m-%d'), # date 객체를 다시 문자열로
                'Dividend': dividend_rate, # 개별 배당금
                'Annual Dividend': annual_dividend, # 연간 배당금
                'Dividend Yield': f"{dividend_yield:.2f}%",
                'Dividend_Yield_Value': dividend_yield, # 정렬을 위한 float 값
                'Payment Date': stock_data['payment_Date'],
                'Current Price': f"${price:.2f}"
            }
        else:
            # 수익률 기준 미달 종목 로깅 (DIVIDEND_YIELD_THRESHOLD 값과 실제 수익률 포함)
            logging.info(f"수익률 기준 미달 ({ticker}, {DIVIDEND_YIELD_THRESHOLD=}%): {dividend_yield:.2f}%")
            return None
            
    except ValueError as e: # float 변환 등에서 오류 발생 시
        logging.error(f"데이터 타입 오류 ({ticker}): {str(e)}")
        return None
    except Exception as e: # 기타 예외 처리
        logging.error(f"종목 처리 중 예기치 않은 오류 ({ticker}): {str(e)}")
        return None # 함수 상단 try-except 제거하고 여기서 None 반환

# --- process_stock 리팩토링 종료 ---

# --- main 함수 리팩토링 시작 ---

def _build_telegram_message_for_chunk(chunk_stocks, target_date, current_time_str, global_start_idx, total_stocks_count):
    """텔레그램으로 보낼 메시지의 특정 청크를 생성합니다.

    Args:
        chunk_stocks (list): 현재 청크에 포함된 주식 정보 딕셔너리의 리스트.
        target_date (str): 조회 대상 날짜 (YYYY-MM-DD 형식).
        current_time_str (str): 현재 시간 문자열 (YYYY-MM-DD HH:MM 형식).
        global_start_idx (int): 현재 청크의 첫 번째 주식에 대한 전체 목록에서의 시작 번호.
        total_stocks_count (int): 필터링된 전체 주식의 수.

    Returns:
        str: 생성된 텔레그램 메시지 문자열.
    """
    message_header = (
        f"<b>[{target_date}] 미국주식 고배당 종목 알림 ({total_stocks_count}건)</b>\n"
        f"※ 동부시간 기준 {current_time_str}\n"
        f"※ 최종 매수 기한: 배당락일 당일 00:00 ET 기준\n\n"
    )

    message_body = ""
    # global_start_idx를 사용하여 각 항목에 대해 올바른 번호 매기기
    for idx, stock_item in enumerate(chunk_stocks, start=global_start_idx):
        # 최종 매수일 계산: 배당락일 1영업일 전 (미국 공휴일 반영)
        ex_date_dt = pd.to_datetime(stock_item['Ex-Date'])
        # APP_HOLIDAYS 리스트를 사용하여 영업일 계산
        cutoff_date = (ex_date_dt - CustomBusinessDay(1, holidays=APP_HOLIDAYS)).strftime('%Y-%m-%d')

        message_body += (
            f"<b>{idx}. {stock_item['Symbol']}</b> ({stock_item['Name']})\n"
            f"├ 배당락일: {stock_item['Ex-Date']}\n"
            f"├ 최종 매수일: {cutoff_date}\n"
            f"├ 현재 가격: {stock_item['Current Price']}\n"
            f"├ 배당 수익률: {stock_item['Dividend Yield']}\n"
            f"├ 배당금: ${stock_item['Dividend']} (연간 ${stock_item['Annual Dividend']})\n"
            f"└ 지급일: {stock_item['Payment Date']}\n\n"
        )
    return message_header + message_body

def main():
    """스크립트의 메인 실행 함수.
    NASDAQ에서 배당 정보를 가져와 필터링하고, 조건에 맞는 종목을 텔레그램으로 알립니다.
    """
    try:
        logging.info("스크립트 실행 시작")
        # 현재 동부시간 기준 날짜 및 시간 정보
        now_et = datetime.now(EASTERN_TZ)
        current_date = now_et.date() # process_stock에서 현재는 미사용
        current_time_str = now_et.strftime('%Y-%m-%d %H:%M')
        
        # 2영업일 후 날짜 계산 (공휴일 반영)
        logging.info("대상 날짜 계산 시작...")
        target_date_str = calculate_target_date()
        if not target_date_str:
            # calculate_target_date 내부에서 이미 로깅됨
            raise ValueError("목표일 계산 실패 (calculate_target_date)")
        logging.info(f"대상 날짜 계산 완료: {target_date_str}")

        # NASDAQ 데이터 조회
        logging.info(f"NASDAQ 배당 정보 조회 시작 (대상 날짜: {target_date_str})...")
        nasdaq_data = fetch_nasdaq_data(target_date_str)
        if not nasdaq_data or not nasdaq_data.get('data'):
            # fetch_nasdaq_data 내부에서 이미 로깅됨
            raise RuntimeError("배당 데이터를 가져오지 못했습니다 (fetch_nasdaq_data).")

        raw_dividend_items = nasdaq_data.get('data', {}).get('calendar', {}).get('rows', [])
        logging.info(f"NASDAQ에서 {len(raw_dividend_items)}개의 배당 정보를 가져왔습니다.")

        # API 응답에서 실제 종목 리스트 추출
        stock_list_from_api = raw_dividend_items # 이미 추출됨
        if not stock_list_from_api:
            logging.info("NASDAQ API에서 반환된 종목이 없습니다. (데이터는 있으나 rows가 비어 있음)")
            logging.info("스크립트 실행 완료")
            return # 정상 종료

        # ThreadPoolExecutor를 사용하여 병렬로 종목 처리
        logging.info("개별 종목 정보 처리 및 필터링 시작...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            # process_stock 함수에 current_date를 전달하나, 현재 내부 로직에서 직접 사용하지는 않음
            results = executor.map(lambda s: process_stock(s, current_date), stock_list_from_api)
            filtered_stocks = [res for res in results if res] # None이 아닌 결과만 필터링

        logging.info(f"총 {len(stock_list_from_api)}개 NASDAQ 항목 중 {len(filtered_stocks)}개 종목이 필터링 기준을 통과했습니다.")

        if not filtered_stocks:
            logging.info("조건에 맞는 배당주 종목이 없습니다. (필터링 후 비어 있음)")
            logging.info("스크립트 실행 완료")
            return # 정상 종료

        # 배당 수익률 높은 순으로 정렬
        filtered_stocks.sort(key=lambda x: x['Dividend_Yield_Value'], reverse=True)

        # 필터링된 종목들을 정해진 크기(chunk_size)로 나누어 텔레그램 메시지 전송
        chunk_size = 10  # 한 번에 보낼 메시지에 포함될 종목 수
        total_stocks_count = len(filtered_stocks)
        total_parts = (total_stocks_count + chunk_size - 1) // chunk_size # 전체 메시지 파트 수

        logging.info(f"Telegram 메시지 전송 시작 ({total_parts}개 파트 예정)...")
        for part_index in range(total_parts):
            chunk_start_offset = part_index * chunk_size
            chunk_end_offset = chunk_start_offset + chunk_size
            current_stock_chunk = filtered_stocks[chunk_start_offset:chunk_end_offset]
            
            # 메시지 본문 생성 (새로운 헬퍼 함수 사용)
            # global_start_idx는 1부터 시작하는 번호
            message_chunk = _build_telegram_message_for_chunk(
                current_stock_chunk,
                target_date_str,
                current_time_str,
                global_start_idx=chunk_start_offset + 1,
                total_stocks_count=total_stocks_count
            )

            if not send_telegram(message_chunk):
                # send_telegram 내부에서 이미 로깅됨
                logging.error(f"Telegram 메시지 전송 실패 (파트 {part_index + 1}/{total_parts}). 프로그램 중단.")
                sys.exit(1) # 중요한 알림 실패 시 스크립트 중단

        logging.info(f"총 {total_stocks_count}개 종목, {total_parts}개 메시지로 전송 완료.")
        logging.info("스크립트 실행 완료")

    except ValueError as ve: # 예상된 오류 (예: 날짜 계산 실패)
        logging.error(f"실행 중 설정 또는 데이터 오류: {str(ve)}")
        sys.exit(1)
    except RuntimeError as re: # 예상된 오류 (예: API 데이터 가져오기 실패)
        logging.error(f"실행 중 API 또는 데이터 처리 오류: {str(re)}")
        sys.exit(1)
    except Exception as e: # 기타 모든 예기치 않은 오류
        logging.critical(f"알 수 없는 심각한 오류 발생: {str(e)}", exc_info=True)
        sys.exit(1)

# --- main 함수 리팩토링 종료 ---

if __name__ == "__main__":
    main()
