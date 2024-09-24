import time
import numpy as np
import pyupbit

# API 키 설정
key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# API 키 읽기
with open(key_file_path, 'r') as file:
    access = file.readline().strip()
    secret = file.readline().strip()

# 거래할 암호화폐 종목 설정
tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-EOS", "KRW-ADA", "KRW-DOGE", "KRW-LOOM", "KRW-SHIB", "KRW-DOGE", "KRW-NEO", "KRW-ARDR", "KRW-GAS", "KRW-HBAR", "KRW-STPT", "KRW-UXLINK","KRW-CKB","KRW-IQ","KRW-ZETA"]

# 종목별 거래 상태 관리 딕셔너리 초기화
trade_state = {ticker: {
    "buy1_price": None,
    "buy2_executed": False,
    "buy3_executed": False,
    "buy4_executed": False,
    "buy5_executed": False,
    "buy6_executed": False,
    "buy7_executed": False,
    "sell_executed": False
} for ticker in tickers}

# 사용자 설정 변수들
interval = "minute1"
rsi_period = 14
rsi_threshold = 30
initial_buy_percent = 0.01

profit_threshold = 0.5  # 수익률이 0.5% 이상일 때 매도
loss_threshold_after_final_buy = -2  # 손실률이 -2% 이하일 때 매도

# 추가 매수 조건 설정 (-1% 이상에서만 추가 매수 실행)
additional_buy_conditions = [
    {"trigger_loss": -1, "buy_ratio": 0.01},
    {"trigger_loss": -1, "buy_ratio": 0.015},
    {"trigger_loss": -1, "buy_ratio": 0.02},
    {"trigger_loss": -1, "buy_ratio": 0.025},
    {"trigger_loss": -1, "buy_ratio": 0.03},
    {"trigger_loss": -1, "buy_ratio": 0.035},
    {"trigger_loss": -1, "buy_ratio": 0.04},
]

# 잔고 조회 함수
def get_balance(currency):
    time.sleep(1/30)  # 초당 30회 제한을 준수하기 위해 딜레이 추가
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == currency:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

# 암호화폐 매수 함수
def buy_crypto(ticker, amount):
    time.sleep(1/8)  # 초당 8회 제한을 준수하기 위해 딜레이 추가
    try:
        return upbit.buy_market_order(ticker, amount)
    except Exception as e:
        print(f"{ticker} 매수 오류 발생: {e}")
        return None

# 암호화폐 매도 함수
def sell_crypto(ticker, amount):
    time.sleep(1/8)  # 초당 8회 제한을 준수하기 위해 딜레이 추가
    try:
        return upbit.sell_market_order(ticker, amount)
    except Exception as e:
        print(f"{ticker} 매도 오류 발생: {e}")
        return None

# 현재 가격 조회 함수
def safe_get_current_price(ticker):
    time.sleep(1/30)  # 초당 30회 제한을 준수하기 위해 딜레이 추가
    try:
        price = pyupbit.get_current_price(ticker)
        if price is None:
            print(f"{ticker}의 현재 가격을 가져올 수 없습니다.")
            return None
        return price
    except Exception as e:
        print(f"{ticker}의 현재 가격 조회 중 오류 발생: {e}")
        return None

# 안전한 OHLCV 데이터 가져오기 함수 (1분봉 데이터)
def get_safe_ohlcv(ticker, interval="minute1", count=15):
    time.sleep(1/30)  # 초당 30회 제한을 준수하기 위해 딜레이 추가
    try:
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
        if df is not None and not df.empty:
            return df
        print(f"{ticker} 데이터 없음, 재시도 중...")
    except Exception as e:
        print(f"{ticker} 데이터 가져오기 오류 발생: {e}")
    return None

# RSI 계산 함수
def calculate_rsi(ticker, period=rsi_period, interval=interval):
    df = get_safe_ohlcv(ticker, interval=interval, count=period + 1)
    if df is None:
        print(f"{ticker}의 {interval} 데이터를 가져올 수 없습니다.")
        return None

    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# RSI 매수 신호 감지 함수
def detect_rsi_buy_signal(ticker, rsi_threshold=rsi_threshold, interval=interval):
    rsi = calculate_rsi(ticker, interval=interval)
    if rsi is None:
        return False

    if rsi <= rsi_threshold:
        print(f"{ticker}: RSI 매수 신호 감지 - RSI 값: {rsi:.2f}")
        return True
    else:
        print(f"{ticker}: RSI 매수 신호 미충족 - RSI 값: {rsi:.2f}")
        return False

# RSI 매수 신호 감지 후 초기 매수 실행 함수
def execute_buy_on_rsi_signal(ticker):
    if detect_rsi_buy_signal(ticker):
        buy_amount = krw_balance * initial_buy_percent
        print(f"{ticker} RSI 매수 신호 감지: {buy_amount} KRW 매수 진행")
        buy_order = buy_crypto(ticker, buy_amount)
        if buy_order is not None:
            trade_state[ticker]['buy1_price'] = safe_get_current_price(ticker)
            print(f"{ticker} RSI 매수 완료: 가격: {trade_state[ticker]['buy1_price']} KRW")
        time.sleep(3)

# 단일 티커에 대한 매수/매도 로직 실행 함수
def trade_single_ticker(ticker):
    current_price = safe_get_current_price(ticker)
    if current_price is None:
        return

    buy1_price = trade_state[ticker]['buy1_price']
    if buy1_price is not None:
        profit_rate = ((current_price - buy1_price) / buy1_price) * 100

        # 매도 조건 1: 수익률이 설정된 값 이상일 때 전체 매도 (0.3% 이상)
        if profit_rate >= profit_threshold and not trade_state[ticker]['sell_executed']:
            trade_state[ticker]['sell_executed'] = True
            crypto_balance = get_balance(ticker.split("-")[1])
            sell_order = sell_crypto(ticker, crypto_balance)
            if sell_order is not None:
                print(f"{ticker} 전액 매도 완료: 수익률 {profit_rate:.2f}%")
                trade_state[ticker]['sell_executed'] = False
            time.sleep(3)

        # 매도 조건 2: 최종 매수 이후 손실률이 설정된 값 이하일 때 매도 (-2% 이하)
        if trade_state[ticker]['buy7_executed'] and profit_rate <= loss_threshold_after_final_buy and not trade_state[ticker]['sell_executed']:
            trade_state[ticker]['sell_executed'] = True
            crypto_balance = get_balance(ticker.split("-")[1])
            sell_order = sell_crypto(ticker, crypto_balance)
            if sell_order is not None:
                print(f"{ticker} 전액 매도 완료: 손실률 {profit_rate:.2f}% (최종 매수 이후)")
                trade_state[ticker]['sell_executed'] = False
            time.sleep(3)

        # 추가 매수 조건 (-1% 이하일 때 추가 매수 실행)
        for i, condition in enumerate(additional_buy_conditions):
            if (profit_rate <= condition['trigger_loss'] and detect_rsi_buy_signal(ticker)
                    and not trade_state[f'buy{i+2}_executed'] and i < 7):
                trade_state[f'buy{i+2}_executed'] = True
                buy_amount = (krw_balance * condition['buy_ratio'])
                buy_order = buy_crypto(ticker, buy_amount)
                if buy_order is not None:
                    krw_balance -= buy_amount
                    print(f"{ticker} 추가 매수 {i+2} 완료: {buy_amount} KRW 매수")
                time.sleep(3)

# 업비트 로그인
upbit = pyupbit.Upbit(access, secret)
print("자동 거래 시작")

# Step 1: 초기 구매 결정
krw_balance = get_balance("KRW")

# Step 2: 각 티커에 대해 1분마다 거래 실행
while True:
    start_time = time.time()
    
    for ticker in tickers:
        # 각 티커의 1분봉 데이터를 가져오고 매매 신호를 확인
        print(f"{ticker}의 데이터를 확인합니다.")
        execute_buy_on_rsi_signal(ticker)
        trade_single_ticker(ticker)
        time.sleep(1)  # 각 종목 처리 후 1초 대기 (API 호출 제한 준수)

    # 1분이 지나기 전까지 대기 (60초에서 남은 시간을 대기)
    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        time.sleep(60 - elapsed_time)
