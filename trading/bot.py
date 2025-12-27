import time
import sys
import os
import datetime
import pybitflyer
import math

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
        
        # --- test.py のロジック用状態変数 ---
        self.exit_countdown = 0
        self.consecutive_losses = 0
        self.pause_remaining = 0  # 取引停止の残り回数 (時間)
        self.peak_portfolio = 0.0
        self.initial_balance = 0.0
        self.first_run = True

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
            trade_eth_balance = max(0.0, eth_balance - 1.0)
            
            # print(f"   実際のETH残高: {eth_balance:.8f} ETH (取引可能: {trade_eth_balance:.8f} ETH)")

            return jpy_balance, trade_eth_balance
        except Exception as e:
            print(f"資産状況の取得中に予期せぬエラーが発生しました: {e}")
            return 0.0, 0.0

    def send_market_order(self, side, size):
        """成行注文を送信する"""
        size = round(size, 8)
        print(f"   注文内容: {side} {size} ETH")
        
        if config.DRY_RUN:
            print("   [DRY RUN] 注文は送信されませんでした。")
            return "dry_run_id"

        try:
            order_res = self.api.sendchildorder(
                product_code="ETH_JPY",
                child_order_type="MARKET",
                side=side,
                size=size
            )
            print(f"   ✅ {side}注文を送信しました。注文ID: {order_res.get('child_order_acceptance_id')}")
            if order_res.get('child_order_acceptance_id'):
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

    def run_trading_logic(self):
        """実際の取引ロジックを実行する (test.py準拠)"""
        print("\n" + "="*50)
        print(f"🤖 ETH自動取引ボット実行中 (本番) ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        # --- 1. 資産と価格の取得 ---
        jpy, eth = self.get_balance()
        current_price = get_ticker()

        if current_price is None:
            return
        
        print(f"   現在のJPY資産: {jpy:,.0f} JPY")
        print(f"   現在のETH残高: {eth:.4f} ETH")
        print(f"   現在のETH価格: {current_price:,.0f} JPY")

        # ポートフォリオ価値計算
        current_portfolio = jpy + (eth * current_price)

        # --- 初回実行時の初期化 ---
        if self.first_run:
            self.initial_balance = current_portfolio
            self.peak_portfolio = current_portfolio
            self.first_run = False
            
            # 既存ポジションがある場合の復旧（簡易判定）
            if eth > 0.005: 
                 self.position = 'long'
                 if self.entry_price is None:
                     self.entry_price = current_price 
                 print("   🔄 既存ポジションを検出しました。監視を継続します。")
            else:
                self.position = 'none'
            
            print(f"   🏁 初期資産設定: {self.initial_balance:,.0f} JPY")

        # --- ドローダウン更新 ---
        if current_portfolio > self.peak_portfolio:
            self.peak_portfolio = current_portfolio
        
        drawdown = 0.0
        if self.peak_portfolio > 0:
            drawdown = (self.peak_portfolio - current_portfolio) / self.peak_portfolio

        portfolio_ratio = 0.0
        if self.initial_balance > 0:
            portfolio_ratio = current_portfolio / self.initial_balance

        # 取引停止期間の更新
        if self.pause_remaining > 0:
            self.pause_remaining -= 1

        print(f"   📊 ポートフォリオ: {current_portfolio:,.0f} JPY (DD: {drawdown:.2%}, Ratio: {portfolio_ratio:.2%})")
        print(f"   ⚠️ 連続損失: {self.consecutive_losses}回, 停止残り: {self.pause_remaining}回")

        # --- 2. 売買判断 ---
        buy = False
        sell = False
        should_buy = False

        # A. ポジション保有時の決済判定
        if self.position == 'long':
            # 1a. 利確決済
            if self.entry_price and current_price > self.entry_price * (1 + config.TAKE_PROFIT_THRESHOLD):
                print(f"   🎉 利確条件達成 (現在: {current_price} > 取得: {self.entry_price} * {1+config.TAKE_PROFIT_THRESHOLD:.2f})")
                sell = True

            # 1b. 損切り決済
            elif self.entry_price and current_price < self.entry_price * (1 - config.STOP_LOSS_THRESHOLD):
                print(f"   😭 損切り条件達成 (現在: {current_price} < 取得: {self.entry_price} * {1-config.STOP_LOSS_THRESHOLD:.2f})")
                sell = True
            
            # 1c. 時間経過決済
            elif self.exit_countdown <= 0:
                print(f"   ⏰ 保持期間終了")
                sell = True
            
            else:
                self.exit_countdown -= 1
                print(f"   ⏳ 決済待機中 (残り {self.exit_countdown} 時間)")

            # 決済条件を満たさない場合、予測処理をスキップして終了 (test.py準拠)
            if not sell:
                print("   🧘 決済条件未達のため、ポジションを継続します。")
                return

        # ここに来るのは position='none' または position='long' and sell=True の場合
        
        # 予測実行
        result = self.predict()
        if result:
            result_class, result_confidence = result
            if result_class == "up" and result_confidence >= config.CONFIDENCE_THRESHOLD:
                should_buy = True

        # B. 継続判定 (position='long' and sell=True)
        if self.position == 'long' and sell:
            if should_buy:
                print(f"   🔄 買いシグナル継続のため、決済をキャンセルしポジションを維持します。")
                print(f"      (基準価格更新: {self.entry_price} -> {current_price}, 期間リセット)")
                self.entry_price = current_price
                self.exit_countdown = config.HOLD_PERIOD
                sell = False # 売却キャンセル

        # C. 新規エントリー判定 (position='none')
        if self.position == 'none' and not sell:
            if should_buy:
                if self.pause_remaining > 0:
                    print(f"   🚫 取引停止期間中のため、エントリーを見送ります (残り {self.pause_remaining} 回)")
                else:
                    buy = True
            else:
                print("   👀 様子見します")

        # --- 3. 注文実行 ---
        
        # 売却処理
        if sell:
            sell_size = eth
            if sell_size >= self.min_order_eth:
                print("\n[判断] 売却を実行します。")
                order = self.send_market_order("SELL", sell_size)
                if order:
                    self.position = 'none'
                    # 勝敗判定
                    if self.entry_price:
                        # 手数料(0.2%程度)を考慮して、0.996倍より高ければ勝ちとみなす簡易判定
                        if current_price * 0.996 > self.entry_price: 
                            print("   ✅ トレード勝利！連続損失リセット")
                            self.consecutive_losses = 0
                        else:
                            print("   ❌ トレード敗北...")
                            self.consecutive_losses += 1
                            if self.consecutive_losses >= 3:
                                self.pause_remaining = 12
                                print("   🛑 3連続損失のため、12時間取引を停止します。")
                    
                    self.entry_price = None
                    self.exit_countdown = 0
            else:
                print(f"   ⚠️ 売却しようとしましたが、残高({sell_size})が最小注文数量未満です。")
                self.position = 'none'
                self.entry_price = None

        # 購入処理
        if buy:
            # ポジションサイズ計算 (test.py準拠)
            position_multiplier = 0.8
            if drawdown > config.MAX_DRAWDOWN_THRESHOLD or self.consecutive_losses >= 3:
                position_multiplier = 0.7
            elif portfolio_ratio < config.MIN_PORTFOLIO_RATIO:
                position_multiplier = 0.6
            
            target_buy_jpy = jpy * position_multiplier
            # 手数料分を引いて購入数量を計算
            buy_size = (target_buy_jpy * (1 - config.FEE_RATE)) / current_price
            
            print(f"   💰 購入計画: 資産の{position_multiplier*100:.0f}% ({target_buy_jpy:,.0f} JPY) を使用")

            if buy_size >= self.min_order_eth:
                print("\n[判断] 購入を実行します。")
                order = self.send_market_order("BUY", buy_size)
                if order:
                    self.position = 'long'
                    self.entry_price = current_price
                    self.exit_countdown = config.HOLD_PERIOD
            else:
                print(f"   ⚠️ 購入可能数量({buy_size:.4f} ETH)が最小注文数量未満のため見送ります。")

    def run(self):
        """メインループ"""
        print("🚀 ボットを起動します...")
        
        while True:
            try:
                self.run_trading_logic()
                
                # 1時間待機
                print(f"\n🕒 次のチェックまで 1時間 待機します...")
                time.sleep(3600)
                
            except FileNotFoundError as e:
                print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py などを実行してモデルを学習してください。")
                break
            except KeyboardInterrupt:
                print("\n👋 ボットを停止します。")
                break
            except Exception as e:
                print(f"予期せぬエラーが発生しました: {e}")
                print("10分後に再試行します...")
                time.sleep(600)
