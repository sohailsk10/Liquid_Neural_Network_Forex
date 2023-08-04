import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta
import pandas as pd
import psycopg2

currency_ = "GBPUSD"

def get_currency():
    currency = currency_
    return currency

def get_live_data(currency):
    try:
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            return None
        else:
            ask = round(mt5.symbol_info_tick(currency).ask, 5)
            bid = round(mt5.symbol_info_tick(currency).bid, 5)
            get_current_date = datetime.now()
            return bid, ask,get_current_date
    except:
        pass


def get_db_conn():
    conn = psycopg2.connect(database="forex_db",
                             user='postgres', 
                             password='admin', 
                             host='127.0.0.1', 
                             port='5432')

    return conn

