import time
import sys
import os
import datetime
import pybitflyer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.btc_data import get_btc_data, create_features
from predictor import predict_class, load_checkpoint
import config

def get_api():
    try:
        if not config.BITFLYER_API_KEY or config.BITFLYER_API_KEY == "BITFLYER_API_KEY":
             raise ValueError("APIキーがconfig.pyに設定されていません。")
        api = pybitflyer.API(api_key=config.BITFLYER_API_KEY, api_secret=config.BITFLYER_API_SECRET)
        return api
    except Exception as e:
        print(f"❌ APIキーの初期化中にエラーが発生しました: {e}")
        print("   config.pyのBITFLYER_API_KEYとBITFLYER_API_SECRETが正しく設定されているか確認してください。")
        sys.exit(1)


def get_ticker():
    """現在のETH価格を取得する"""
    try:
        public_api = pybitflyer.API()
        ticker = public_api.ticker(product_code="ETH_JPY")
        return ticker['ltp']
    except Exception as e:
        print(f"価格の取得に失敗しました: {e}")
        return None


class TradingBot:
    def __init__(self):
        self.api = get_api()
        self.min_order_eth = 0.01
        self.entry_price = None  # 購入価格をメモリ上に保持
        self.position = "none" # 'long' or 'none'
        self.model, self.scaler = load_checkpoint()
        self.exit_countdown = 0

    def get_balance(self) -> tuple[float, float]:
        """資産状況を取得する。保有している1.0 ETHを無視するロジックを含む。"""
        try:
            balances = self.api.getbalance()
            if not isinstance(balances, list):
                print(f"❌ 資産状況の取得に失敗しました。APIからの応答: {balances}")
                print("   APIキーが正しいか、Bitflyerのステータスを確認してください。")
                return 0.0, 0.0

            jpy_balance = 0.0
            eth_balance = 0.0
            for balance in balances:
                if balance['currency_code'] == 'JPY':
                    jpy_balance = float(balance['available'])
                elif balance['currency_code'] == 'ETH':
                    eth_balance = float(balance['available'])
            
            # --- ユーザー要望: 1.0 ETHを取引対象外とする ---
            # 実際のETH残高から1.0を差し引いた値を、取引判断に利用する残高とする。
            # これにより、元々保有している1.0 ETHは売買されなくなる。
            trade_eth_balance = max(0.0, eth_balance - 1.0)
            
            print(f"   実際のETH残高: {eth_balance:.8f} ETH")

            return jpy_balance, trade_eth_balance
        except Exception as e:
            print(f"資産状況の取得中に予期せぬエラーが発生しました: {e}")
            return 0.0, 0.0

    def send_market_order(self, side, size):
        """成行注文を送信する"""
        size = round(size, 8)
        print(f"   注文内容: {side} {size} ETH")
        
        try:
            order_res = self.api.sendchildorder(
                product_code="ETH_JPY",
                child_order_type="MARKET",
                side=side,
                size=size
            )
            print(f"   ✅ {side}注文を送信しました。注文ID: {order_res['child_order_acceptance_id']}")
            if order_res['child_order_acceptance_id']:
                return order_res['child_order_acceptance_id']
            else:
                print("   ⚠️ 注文情報を取得できませんでした。")
                return None
        except Exception as e:
            print(f"   ❌ {side}注文の送信中にエラーが発生しました: {e}")
            return None

    def predict(self):
        df = get_btc_data(period="5d", interval="1h")
        if df.empty:
            print("   データ取得に失敗しました。処理を中断します。")
            return None
        df_with_features = create_features(df)
        features = df_with_features[config.FEATURE_COLUMNS].values

        # --- 予測実行 ---
        if len(features) < config.L:
            print(f"   予測に必要なデータが不足しています。現在 {len(features)} 個, 必要なのは {config.L} 個。")
            return None

        features_seq = features[-config.L:]
        result = predict_class(self.model, self.scaler, features_seq)

        print(f"\n🧠 モデル予測結果:")
        print(f"   予測: {result['class']} | 信頼度: {result['confidence']:.2%}")

        return result['class'], result['confidence']

    def restore_position(self):
        jpy, _eth = self.get_balance()
        if jpy >= 500:
            self.position = 'none'
        else:
            self.position = 'long'

    def run_trading_logic(self):
        """実際の取引ロジックを実行する"""
        print("\n" + "="*50)
        print(f"🤖 ETH自動取引ボット実行中 (本番) ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        # --- 初期状態確認 ---
        jpy, eth = self.get_balance()

        print(f"   現在のJPY資産: {jpy:,.0f} JPY")

        current_price = get_ticker()

        if current_price is None:
            return
        print(f"   現在のETH価格: {current_price:,.0f} JPY")

        result = self.predict()

        if result is None:
            return

        result_class, result_confidence = result
        buy = False
        sell = False

        # --- 1. 決済の確認 ---
        if self.position == 'long':
            # 1a. 損切り決済
            if self.entry_price is not None: # 再起動後longで開始の場合はentry_priceがNoneになる
                stop_loss_price = self.entry_price * (1 - config.STOP_LOSS_THRESHOLD)

                if current_price < stop_loss_price:
                    print(f"   ⚠️ 現在価格が損切りラインを下回っています！")
                    sell = True

            self.exit_countdown -= 1

            # 1b. 時間経過による決済
            if self.exit_countdown == 0:
                sell = True
            else:
                print("\n🧘 [判断] 売却条件を満たさなかったため、待機します。")

        if sell:
            self.position = 'none'

        if self.position == 'none':
            if result_class == "up" and result_confidence >= config.CONFIDENCE_THRESHOLD:
                buy = True

        if buy and sell:
            # 同時に売り買い＝何もしない（longのまま）
            self.position = 'long'
            self.entry_price = current_price
            self.exit_countdown = config.HOLD_PERIOD
        else:
            if buy:
                buy_size = (jpy * (1 - config.FEE_RATE)) / current_price
                if buy_size >= self.min_order_eth:
                    print("\n[判断] 購入条件を満たしました。")
                    order = self.send_market_order("BUY", buy_size)
                    if order:
                        self.position = 'long'
                        self.entry_price = current_price
                        self.exit_countdown = config.HOLD_PERIOD

            if sell:
                sell_size = eth
                if sell_size >= self.min_order_eth:
                    print("\n[判断] 売却条件を満たしました。")
                    order = self.send_market_order("SELL", sell_size)
                    if order:
                        self.entry_price = None
                        self.position = 'none'

    def run(self):
        """メインループ"""
        self.restore_position()
        self.run_trading_logic()

        while True:
            try:
                print(f"\n🕒 次の実行まで {config.HOLD_PERIOD} 時間待機します...")
                time.sleep(config.HOLD_PERIOD * 60 * 60)
                self.run_trading_logic()
            except FileNotFoundError as e:
                print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py などを実行してモデルを学習してください。")
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"予期せぬエラーが発生しました: {e}")
                print("10分後に再試行します...")
                time.sleep(600)
