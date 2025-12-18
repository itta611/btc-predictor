import pybitflyer
import time
import sys
import os
import numpy as np
import torch
import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.btc_data import get_btc_data, create_features
from predictor import predict_class, load_checkpoint
import config

# Bitflyerの最小注文数量 (BTC)。変更される可能性があるので、BitflyerのAPIドキュメントで確認してください。
MIN_ORDER_BTC = 0.001 

def get_api():
    """Bitflyer APIオブジェクトを取得する"""
    if config.DRY_RUN:
        return None # ドライランモードではAPIオブジェクトは不要
    try:
        api = pybitflyer.API(api_key=config.BITFLYER_API_KEY, api_secret=config.BITFLYER_API_SECRET)
        return api
    except Exception as e:
        print(f"❌ APIキーの初期化中にエラーが発生しました: {e}")
        print("   config.pyのBITFLYER_API_KEYとBITFLYER_API_SECRETが正しく設定されているか確認してください。")
        sys.exit(1)

def get_balance(api):
    """資産状況を取得する"""
    if config.DRY_RUN:
        # ドライラン用のダミーデータを返す
        print("   (DRY RUN) ダミーの資産状況を使用します。")
        return 100000, 0.005 # 10万円, 0.005 BTC
    try:
        balances = api.getbalance()
        jpy_balance = 0
        btc_balance = 0
        for balance in balances:
            if balance['currency_code'] == 'JPY':
                jpy_balance = balance['available']
            elif balance['currency_code'] == 'BTC':
                btc_balance = balance['available']
        return jpy_balance, btc_balance
    except Exception as e:
        print(f"資産状況の取得に失敗しました: {e}")
        return None, None

def get_ticker(api):
    """現在のBTC価格を取得する"""
    # Tickerは市場価格なので、ドライランでも実際の値を取得する
    try:
        # ドライランでも価格は必要なので、キーなしで初期化
        public_api = pybitflyer.API() 
        ticker = public_api.ticker(product_code="BTC_JPY")
        return ticker['ltp']
    except Exception as e:
        print(f"価格の取得に失敗しました: {e}")
        return None

def send_market_order(api, side, size):
    """成行注文を送信する"""
    print(f"   注文内容: {side} {size:.8f} BTC")
    if config.DRY_RUN:
        print("   -> (DRY RUN) 注文は送信されませんでした。")
        return {'status': 'dry_run'}
    
    try:
        order = api.sendchildorder(
            product_code="BTC_JPY",
            child_order_type="MARKET",
            side=side,
            size=size
        )
        if 'child_order_acceptance_id' in order:
            print(f"   ✅ {side}注文を送信しました。注文ID: {order['child_order_acceptance_id']}")
            return order
        else:
            print(f"   ❌ {side}注文の送信に失敗しました: {order}")
            return None
    except Exception as e:
        print(f"   ❌ {side}注文の送信中にエラーが発生しました: {e}")
        return None

def run_trading_logic():
    """実際の取引ロジックを実行する"""
    header = "🤖 BTC自動取引ボット実行中 (DRY RUN)" if config.DRY_RUN else "🤖 BTC自動取引ボット実行中 (本番)"
    print("\n" + "="*50)
    print(f"{header} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*50)

    api = get_api()
    try:
        model, scaler = load_checkpoint()
    except FileNotFoundError:
        print("❌ エラー: モデルファイルが見つかりません。")
        print("💡 解決方法: modeling/btc_train.py を実行してモデルを学習してください。")
        return # エラーで終了

    # --- 初期状態確認 ---
    jpy, btc = get_balance(api)
    if jpy is None:
        return
    print(f"   現在の資産: {jpy:,.0f} JPY, {btc:.8f} BTC")

    # --- データ準備 ---
    df = get_btc_data(period="3d", interval="1h") 
    if df.empty:
        print("   データ取得に失敗しました。処理を中断します。")
        return
    df_with_features = create_features(df)
    features = df_with_features[config.FEATURE_COLUMNS].values

    # --- 予測実行 ---
    if len(features) < config.L:
        print(f"   予測に必要なデータが不足しています。現在 {len(features)} 個, 必要なのは {config.L} 個。")
        return
        
    features_seq = features[-config.L:]
    result = predict_class(model, scaler, features_seq)
    
    print(f"\n🧠 モデル予測結果:")
    print(f"   予測: {result['class']} | 信頼度: {result['confidence']:.2%}")

    # --- 取引判断 ---
    current_price = get_ticker(api)
    if current_price is None:
        return
    print(f"   現在のBTC価格: {current_price:,.0f} JPY")

    has_btc = btc >= MIN_ORDER_BTC 

    if not has_btc:
        if result['class'] == 'up' and result['confidence'] >= config.CONFIDENCE_THRESHOLD:
            print("\n📈 [判断] 購入条件を満たしました。")
            buy_size = (jpy * (1 - config.FEE_RATE)) / current_price
            
            if buy_size >= MIN_ORDER_BTC:
                send_market_order(api, "BUY", buy_size)
            else:
                print(f"   -> 購入可能数量 ({buy_size:.8f} BTC) が最小注文数量 ({MIN_ORDER_BTC} BTC) 未満のため、購入を見送ります。")
        else:
            print("\n🧘 [判断] 購入条件を満たさなかったため、待機します。")
    else:
        if result['class'] == 'Not-Up':
            print("\n📉 [判断] 売却条件を満たしました。")
            sell_size = btc
            
            if sell_size >= MIN_ORDER_BTC:
                send_market_order(api, "SELL", sell_size)
            else:
                print(f"   -> 売却可能数量 ({sell_size:.8f} BTC) が最小注文数量 ({MIN_ORDER_BTC} BTC) 未満のため、売却を見送ります。")
        else:
            print("\n🧘 [判断] 売却条件を満たさなかったため、待機します。")


def main():
    """メインループ"""
    # 最初の実行を即座に行う
    run_trading_logic()

    while True:
        try:
            print(f"\n🕒 次の実行まで {config.HOLD_PERIOD} 時間待機します...")
            time.sleep(config.HOLD_PERIOD * 60 * 60)
            run_trading_logic()
        except FileNotFoundError as e:
            print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py を実行してモデルを学習してください。")
            break
        except KeyboardInterrupt:
            print("\n🛑 ボットを手動で停止しました。")
            break
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
            print("10分後に再試行します...")
            time.sleep(600)

if __name__ == "__main__":
    main()
