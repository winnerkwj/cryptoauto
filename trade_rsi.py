import time
import numpy as np
import pyupbit

# **API 키 설정**
# 업비트 API를 사용하기 위해 파일에서 API 키(액세스 키와 시크릿 키)를 불러옵니다.
# key_file_path는 사용자가 실제로 사용하는 API 키 파일의 경로로 바꿔줘야 합니다.
key_file_path = r'C:\Users\winne\OneDrive\바탕 화면\upbit_key.txt'

# **API 키 읽기**
# 해당 경로에서 파일을 열고, 첫 줄에 있는 액세스 키와 두 번째 줄에 있는 시크릿 키를 읽어옵니다.
with open(key_file_path, 'r') as file:
    access = file.readline().strip()  # 첫 번째 줄에서 Access Key 가져오기
    secret = file.readline().strip()  # 두 번째 줄에서 Secret Key 가져오기

# **거래할 암호화폐 종목 설정**
# 매매할 암호화폐 티커를 리스트로 미리 설정해둡니다. 
tickers = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-EOS", "KRW-ADA", "KRW-DOGE"]

# **종목별 거래 상태 관리 딕셔너리 초기화**
# 각 암호화폐 종목에 대해 매매 상태를 저장하는 딕셔너리를 설정합니다.
# 각 단계별 매수/매도 여부를 추적하기 위해 여러 플래그를 사용합니다.
trade_state = {ticker: {
    "buy1_price": None,       # 첫 매수 가격
    "buy2_executed": False,   # 2단계 매수 실행 여부
    "buy3_executed": False,   # 3단계 매수 실행 여부
    "buy4_executed": False,   # 4단계 매수 실행 여부
    "buy5_executed": False,   # 5단계 매수 실행 여부
    "buy6_executed": False,   # 6단계 매수 실행 여부
    "buy7_executed": False,   # 7단계 매수 실행 여부
    "sell_executed": False    # 매도 실행 여부
} for ticker in tickers}

# **사용자 설정 변수들**
# RSI 전략에서 필요한 설정 값을 지정합니다. 
interval = "minute3"  # 3분봉 데이터를 사용 (다른 주기 예: "minute1", "minute5")
rsi_period = 14  # RSI 계산에 사용할 기간 (14일 또는 14개의 3분봉)
rsi_threshold = 30  # RSI가 30 이하일 때 매수 신호
initial_buy_percent = 0.01  # 첫 매수 시 잔고의 1%를 매수 금액으로 설정

# **매도 조건 설정**
# 수익이 1% 이상일 때 매도하거나, 손실이 최종 매수 후 -3%일 때 매도
profit_threshold = 1  # 수익률이 1% 이상일 때 매도
loss_threshold_after_final_buy = -3  # 최종 매수 이후 손실이 -3%일 때 매도

# **추가 매수 조건 설정**
# 손실률에 따라 추가 매수할 비율을 정의합니다. 손실이 커질수록 더 많은 양을 추가 매수합니다.
additional_buy_conditions = [
    {"trigger_loss": -1, "buy_ratio": 0.01},    # 손실률 -2%일 때 추가 매수 1%
    {"trigger_loss": -4, "buy_ratio": 0.015},   # 손실률 -4%일 때 추가 매수 1.5%
    {"trigger_loss": -6, "buy_ratio": 0.02},    # 손실률 -6%일 때 추가 매수 2%
    {"trigger_loss": -8, "buy_ratio": 0.025},   # 손실률 -8%일 때 추가 매수 2.5%
    {"trigger_loss": -10, "buy_ratio": 0.03},   # 손실률 -10%일 때 추가 매수 3%
    {"trigger_loss": -12, "buy_ratio": 0.035},  # 손실률 -12%일 때 추가 매수 3.5%
    {"trigger_loss": -15, "buy_ratio": 0.04},   # 손실률 -15%일 때 추가 매수 4%
]

# **잔고 조회 함수**
# 해당 통화(KRW 또는 암호화폐)의 잔고를 조회합니다.
def get_balance(currency):
    balances = upbit.get_balances()  # 업비트 계좌의 잔고를 조회합니다.
    for b in balances:
        if b['currency'] == currency:
            if b['balance'] is not None:
                return float(b['balance'])  # 잔고가 있으면 해당 잔고를 반환
            else:
                return 0  # 잔고가 없으면 0을 반환
    return 0  # 요청한 통화에 대한 정보가 없으면 0 반환

# **암호화폐 매수 함수**
# 지정된 암호화폐를 주어진 금액(KRW 기준)으로 시장가로 매수합니다.
def buy_crypto(ticker, amount):
    try:
        return upbit.buy_market_order(ticker, amount)  # 지정한 금액만큼 시장가로 매수
    except Exception as e:
        print(f"{ticker} 매수 오류 발생: {e}")
        return None

# **암호화폐 매도 함수**
# 지정된 암호화폐를 주어진 수량만큼 시장가로 매도합니다.
def sell_crypto(ticker, amount):
    try:
        return upbit.sell_market_order(ticker, amount)  # 시장가로 매도
    except Exception as e:
        print(f"{ticker} 매도 오류 발생: {e}")
        return None

# **현재 가격 조회 함수**
# 해당 암호화폐의 현재 가격을 조회합니다.
def safe_get_current_price(ticker):
    try:
        price = pyupbit.get_current_price(ticker)  # 현재 시장 가격 조회
        if price is None:
            print(f"{ticker}의 현재 가격을 가져올 수 없습니다.")
            return None
        return price
    except Exception as e:
        print(f"{ticker}의 현재 가격 조회 중 오류 발생: {e}")
        return None

# **RSI 계산 함수**
# 설정된 기간 동안 RSI를 계산합니다.
def calculate_rsi(ticker, period=rsi_period, interval=interval):
    try:
        # 해당 티커의 기간별 데이터를 가져옵니다.
        df = pyupbit.get_ohlcv(ticker, interval=interval, count=period + 1)
        if df is None or df.empty:
            print(f"{ticker}의 {interval} 데이터를 가져올 수 없습니다.")
            return None

        # 종가 차이를 계산합니다.
        delta = df['close'].diff()

        # 상승분과 하락분을 분리하여 계산합니다.
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # 평균 상승과 평균 하락을 계산합니다.
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])

        if avg_loss == 0:
            return 100  # 손실이 없으면 RSI는 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi
    except Exception as e:
        print(f"{ticker}의 {interval} RSI 계산 오류 발생: {e}")
        return None

# **RSI 매수 신호 감지 함수**
# RSI가 설정된 임계치 이하일 때 매수 신호를 감지합니다.
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

# **RSI 매수 신호 감지 후 초기 매수 실행 함수**
# RSI 매수 신호를 감지한 후 초기 매수를 실행합니다.
def execute_buy_on_rsi_signal(ticker):
    if detect_rsi_buy_signal(ticker):
        buy_amount = krw_balance * initial_buy_percent  # 초기 매수 금액 설정
        print(f"{ticker} RSI 매수 신호 감지: {buy_amount} KRW 매수 진행")
        buy_order = buy_crypto(ticker, buy_amount)  # 매수 실행
        if buy_order is not None:
            trade_state[ticker]['buy1_price'] = safe_get_current_price(ticker)  # 첫 매수 가격 기록
            print(f"{ticker} RSI 매수 완료: 가격: {trade_state[ticker]['buy1_price']} KRW")
        time.sleep(3)  # 매수 후 3초 대기

# **업비트 로그인**
# 업비트 API를 통해 거래를 시작합니다.
upbit = pyupbit.Upbit(access, secret)
print("자동 거래 시작")

# **Step 1: 초기 구매 결정**
krw_balance = get_balance("KRW")  # 원화 잔고 조회
jan1_percent = krw_balance * initial_buy_percent  # 초기 매수 금액 설정

# 각 암호화폐에 대해 초기 구매 진행
for ticker in tickers:
    execute_buy_on_rsi_signal(ticker)  # RSI 매수 신호 감지 후 매수 실행

# **Step 3: 수익률 모니터링 및 동적 추가 구매/매도 실행**
while True:
    for ticker in tickers:
        current_price = safe_get_current_price(ticker)  # 현재 가격 조회
        if current_price is None:
            continue

        buy1_price = trade_state[ticker]['buy1_price']  # 첫 매수 가격
        if buy1_price is not None:
            # 현재 가격과 첫 매수 가격을 비교해 수익률 계산
            profit_rate = ((current_price - buy1_price) / buy1_price) * 100

            # 매도 조건 1: 수익률이 설정된 값 이상일 때 전체 매도
            if profit_rate >= profit_threshold and not trade_state[ticker]['sell_executed']:
                trade_state[ticker]['sell_executed'] = True
                crypto_balance = get_balance(ticker.split("-")[1])  # 해당 암호화폐 잔고 조회
                sell_order = sell_crypto(ticker, crypto_balance)  # 전량 매도
                if sell_order is not None:
                    print(f"{ticker} 전액 매도 완료: 수익률 {profit_rate:.2f}%")
                    trade_state[ticker]['sell_executed'] = False  # 다시 매수 가능하도록 초기화
                time.sleep(3)  # 매도 후 3초 대기

            # 매도 조건 2: 최종 매수 이후 손실률이 설정된 값 이하일 때 매도
            if trade_state[ticker]['buy7_executed'] and profit_rate <= loss_threshold_after_final_buy and not trade_state[ticker]['sell_executed']:
                trade_state[ticker]['sell_executed'] = True
                crypto_balance = get_balance(ticker.split("-")[1])  # 잔고 조회
                sell_order = sell_crypto(ticker, crypto_balance)  # 전량 매도
                if sell_order is not None:
                    print(f"{ticker} 전액 매도 완료: 손실률 {profit_rate:.2f}% (최종 매수 이후)")
                    trade_state[ticker]['sell_executed'] = False
                time.sleep(3)

            # 추가 매수 단계별 조건 평가
            for i, condition in enumerate(additional_buy_conditions):
                if (profit_rate <= condition['trigger_loss']) and detect_rsi_buy_signal(ticker) and not trade_state[f'buy{i+2}_executed'] and i < 7:
                    trade_state[f'buy{i+2}_executed'] = True
                    buy_amount = (krw_balance * condition['buy_ratio'])  # 추가 매수 금액 계산
                    buy_order = buy_crypto(ticker, buy_amount)  # 추가 매수 실행
                    if buy_order is not None:
                        krw_balance -= buy_amount  # 매수 후 잔고 업데이트
                        print(f"{ticker} 추가 매수 {i+2} 완료: {buy_amount} KRW 매수")
                    time.sleep(3)

        time.sleep(10)  # 10초마다 가격 확인
