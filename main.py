import requests
from datetime import datetime
import yfinance as yf
import os
import sys

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram(message):
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
        print(f"Telegram 전송 실패: {str(e)}", file=sys.stderr)
        return False

def main():
    today = datetime.now().strftime('%Y-%m-%d')
    
    # NASDAQ API 호출
    url = f'https://api.nasdaq.com/api/calendar/dividends?date={today}'
    headers = {
        'accept': 'application/json, text/plain, */*',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36',
        'referer': 'https://www.nasdaq.com/'
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    filtered_stocks = []
    if data.get('data') and data['data'].get('calendar', {}).get('rows'):
        for stock in data['data']['calendar']['rows']:
            try:
                dividend_rate = float(stock['dividend_Rate'])
                annual_dividend = float(stock['indicated_Annual_Dividend'])
                
                ticker = stock['symbol']
                price = yf.Ticker(ticker).history(period='1d').iloc[-1].Close
                dividend_yield = (annual_dividend / price) * 100 if price else 0
                
                if dividend_yield >= 3:
                    filtered_stocks.append({
                        'Symbol': ticker,
                        'Name': stock['companyName'],
                        'Ex-Date': stock['dividend_Ex_Date'],
                        'Dividend': dividend_rate,
                        'Annual Dividend': annual_dividend,
                        'Dividend Yield': f"{dividend_yield:.2f}%",
                        'Dividend_Yield_Value': dividend_yield,
                        'Payment Date': stock['payment_Date']
                    })
            except Exception as e:
                print(f"종목 처리 오류 {stock.get('symbol')}: {str(e)}", file=sys.stderr)

    filtered_stocks.sort(key=lambda x: x['Dividend_Yield_Value'], reverse=True)

    message = f"[{today}] 배당락일 고배당 종목 ({len(filtered_stocks)}건)\n\n"
    for idx, stock in enumerate(filtered_stocks, 1):
        message += (
            f"[{idx}] {stock['Symbol']} ({stock['Name']})\n"
            f"  ∙ 배당락일: {stock['Ex-Date']}\n"
            f"  ∙ 배당금: ${stock['Dividend']} (연간 ${stock['Annual Dividend']})\n"
            f"  ∙ 배당 수익률: {stock['Dividend Yield']}\n"
            f"  ∙ 지급일: {stock['Payment Date']}\n\n"
        )

    if not send_telegram(message):
        sys.exit(1)

if __name__ == "__main__":
    main()
