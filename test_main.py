import unittest
from unittest.mock import patch, MagicMock
from datetime import date, datetime
import pandas as pd
import pytz # Added import
from pandas.tseries.holiday import USFederalHolidayCalendar

# Assuming main.py is in the same directory or accessible via PYTHONPATH
import main

class TestDividendAlerts(unittest.TestCase):

    def setUp(self):
        # Default configurations for tests
        main.DIVIDEND_YIELD_THRESHOLD = 3.0
        main.NASDAQ_API_URL = 'https://api.nasdaq.com/api/calendar/dividends'
        # Mock EASTERN_TZ for consistent date calculations if needed, though most functions use date objects
        main.EASTERN_TZ = pytz.timezone('US/Eastern') # Requires import pytz in test_main.py

    @patch('main.datetime')
    def test_calculate_target_date(self, mock_datetime):
        # Test normal weekday
        mock_datetime.now.return_value = main.EASTERN_TZ.localize(datetime(2024, 3, 4)) # Monday
        self.assertEqual(main.calculate_target_date(), "2024-03-06") # Wednesday

        # Test across weekend
        mock_datetime.now.return_value = main.EASTERN_TZ.localize(datetime(2024, 3, 8)) # Friday
        self.assertEqual(main.calculate_target_date(), "2024-03-12") # Tuesday (Mon, Tue)

        # Test before a known holiday (e.g., Good Friday - March 29, 2024)
        # USFederalHolidayCalendar includes GoodFriday
        mock_datetime.now.return_value = main.EASTERN_TZ.localize(datetime(2024, 3, 27)) # Wednesday
        # Expect: Thu (28th), Fri (29th - Holiday), Mon (Apr 1st) -> Target: Tue (Apr 2nd)
        # Actually, BusinessDay(2) means 2 non-holiday weekdays *after* current.
        # So, if current is Wed(27th), +1Bday is Thu(28th), +2Bdays is Mon(Apr 1st, as Fri is holiday)
        self.assertEqual(main.calculate_target_date(), "2024-04-01")


    def test_validate_stock_data(self):
        required = ['symbol', 'companyName']
        self.assertTrue(main._validate_stock_data({'symbol': 'T', 'companyName': 'ATT'}, required))
        self.assertFalse(main._validate_stock_data({'symbol': 'T'}, required))
        self.assertFalse(main._validate_stock_data({'symbol': 'T', 'companyName': ''}, required))
        self.assertFalse(main._validate_stock_data({}, required))
        self.assertFalse(main._validate_stock_data(None, required))

    @patch('main.yf.Ticker')
    def test_get_stock_price_info(self, mock_yfinance_ticker):
        mock_ticker_instance = MagicMock()
        mock_yfinance_ticker.return_value = mock_ticker_instance

        # Valid price
        mock_hist_df = pd.DataFrame({'Close': [10.0, 11.0, 12.0, 13.0, 14.0]})
        mock_ticker_instance.history.return_value = mock_hist_df
        self.assertEqual(main._get_stock_price_info('AAPL'), 14.0)

        # Last price is NaN, use previous
        mock_hist_df_nan = pd.DataFrame({'Close': [10.0, 11.0, 12.0, 13.0, pd.NA]})
        mock_ticker_instance.history.return_value = mock_hist_df_nan
        self.assertEqual(main._get_stock_price_info('AAPL'), 13.0)

        # All prices NaN
        mock_hist_df_all_nan = pd.DataFrame({'Close': [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]})
        mock_ticker_instance.history.return_value = mock_hist_df_all_nan
        self.assertIsNone(main._get_stock_price_info('AAPL'))

        # Empty history
        mock_ticker_instance.history.return_value = pd.DataFrame()
        self.assertIsNone(main._get_stock_price_info('MSFT'))

        # yfinance exception
        mock_ticker_instance.history.side_effect = Exception("API error")
        self.assertIsNone(main._get_stock_price_info('GOOG'))


    @patch('main._get_stock_price_info')
    def test_process_stock_logic(self, mock_get_price_info):
        main.DIVIDEND_YIELD_THRESHOLD = 3.0
        sample_stock_data_valid = {
            'symbol': 'TEST', 'companyName': 'Test Inc.', 'dividend_Ex_Date': '03/15/2024',
            'dividend_Rate': '0.5', 'indicated_Annual_Dividend': '2.0', 'payment_Date': '04/01/2024'
        }

        # Test case: Meets yield threshold
        mock_get_price_info.return_value = 50.0 # Yield = (2.0 / 50.0) * 100 = 4.0%
        result = main.process_stock(sample_stock_data_valid.copy(), date(2024, 3, 1))
        self.assertIsNotNone(result)
        self.assertEqual(result['Symbol'], 'TEST')
        self.assertEqual(result['Dividend Yield'], "4.00%")

        # Test case: Below yield threshold
        mock_get_price_info.return_value = 100.0 # Yield = (2.0 / 100.0) * 100 = 2.0%
        result = main.process_stock(sample_stock_data_valid.copy(), date(2024, 3, 1))
        self.assertIsNone(result)

        # Test case: Price info not available
        mock_get_price_info.return_value = None
        result = main.process_stock(sample_stock_data_valid.copy(), date(2024, 3, 1))
        self.assertIsNone(result)

        # Test case: Missing required fields
        invalid_data_missing = sample_stock_data_valid.copy()
        del invalid_data_missing['dividend_Rate']
        result = main.process_stock(invalid_data_missing, date(2024, 3, 1))
        self.assertIsNone(result)

        # Test case: Invalid ex_dividend_date format
        invalid_data_date = sample_stock_data_valid.copy()
        invalid_data_date['dividend_Ex_Date'] = 'INVALID_DATE'
        result = main.process_stock(invalid_data_date, date(2024, 3, 1))
        self.assertIsNone(result)

        # Test case: Indicated annual dividend is zero
        zero_dividend_data = sample_stock_data_valid.copy()
        zero_dividend_data['indicated_Annual_Dividend'] = '0.0'
        mock_get_price_info.return_value = 50.0 # Price doesn't matter if dividend is 0
        result = main.process_stock(zero_dividend_data, date(2024, 3, 1))
        self.assertIsNone(result) # Yield will be 0, less than 3%

        # Test case: Price is zero (should handle ZeroDivisionError or result in 0 yield)
        price_zero_data = sample_stock_data_valid.copy()
        mock_get_price_info.return_value = 0.0
        result = main.process_stock(price_zero_data, date(2024, 3, 1))
        # Depending on implementation, could be None (if yield is 0 then < threshold) or specific error handling
        # Current main.py calculates yield as 0.0 if price is 0, so it will be None.
        self.assertIsNone(result)


    def test_build_telegram_message_for_chunk(self):
        sample_stocks_chunk = [
            {
                'Symbol': 'T1', 'Name': 'TestCo1', 'Ex-Date': '2024-03-15', # Friday
                'Current Price': '$100.00', 'Dividend Yield': '5.00%',
                'Dividend': 1.25, 'Annual Dividend': 5.00, 'Payment Date': '2024-04-01'
            },
            {
                'Symbol': 'T2', 'Name': 'TestCo2', 'Ex-Date': '2024-03-18', # Monday
                'Current Price': '$50.00', 'Dividend Yield': '6.00%',
                'Dividend': 0.75, 'Annual Dividend': 3.00, 'Payment Date': '2024-04-05'
            }
        ]
        target_date = "2024-03-18" # Example, not directly used in message header in this test
        current_time_str = "2024-03-13 10:00"

        message = main._build_telegram_message_for_chunk(sample_stocks_chunk, target_date, current_time_str, 1, 2)

        self.assertIn("<b>[2024-03-18] 미국주식 고배당 종목 알림 (2건)</b>", message)
        self.assertIn("※ 동부시간 기준 2024-03-13 10:00", message)

        # Test stock 1 details (Ex-Date: Mar 15, Fri -> Cutoff: Mar 14, Thu)
        self.assertIn("<b>1. T1</b> (TestCo1)", message)
        self.assertIn("├ 배당락일: 2024-03-15", message)
        self.assertIn("├ 최종 매수일: 2024-03-14", message) # 1 BDay before Fri Mar 15 is Thu Mar 14
        self.assertIn("├ 현재 가격: $100.00", message)

        # Test stock 2 details (Ex-Date: Mar 18, Mon -> Cutoff: Mar 15, Fri)
        self.assertIn("<b>2. T2</b> (TestCo2)", message)
        self.assertIn("├ 배당락일: 2024-03-18", message)
        self.assertIn("├ 최종 매수일: 2024-03-15", message) # 1 BDay before Mon Mar 18 is Fri Mar 15

if __name__ == '__main__':
    # Need to add pytz for EASTERN_TZ to be available in main module during test
    import pytz
    unittest.main()
