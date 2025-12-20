import pybitflyer
import time
import sys
import os
import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ユーザー注記: get_btc_dataという名前ですが、中身はETHデータを取得するように変更されている前提です
from utils.btc_data import get_btc_data, create_features
from predictor import predict_class, load_checkpoint
import config

# BitflyerのETHの最小注文数量。
MIN_ORDER_ETH = 0.01

def get_api():
    """Bitflyer APIオブジェクトを取得する"""
    if config.DRY_RUN:
        return None
    try:
        if not config.BITFLYER_API_KEY or config.BITFLYER_API_KEY == "BITFLYER_API_KEY":
             raise ValueError("APIキーがconfig.pyに設定されていません。")
        api = pybitflyer.API(api_key=config.BITFLYER_API_KEY, api_secret=config.BITFLYER_API_SECRET)
        return api
    except Exception as e:
        print(f"❌ APIキーの初期化中にエラーが発生しました: {e}")
        print("   config.pyのBITFLYER_API_KEYとBITFLYER_API_SECRETが正しく設定されているか確認してください。")
        sys.exit(1)

def get_balance(api):
    """資産状況を取得する。保有している1.0 ETHを無視するロジックを含む。"""
    if config.DRY_RUN:
        print("   (DRY RUN) ダミーの資産状況を使用します (ETH残高は0として扱います)。")
        # ドライランでもユーザーの状況に合わせて残高0でシミュレーション
        # テストのために、もしtrade_state_dry.jsonがあればETHを持っていることにするロジックを入れてもいいが、
        # ここではシンプルに0を返す（買い注文のテスト用）
        # ただし、売り注文（損切り）をテストしたい場合はここを手動で書き換えるか、
        # load_entry_price()の結果を見て分岐する必要があるかもしれない。
        # 今回はシンプルに実装する。
        return 100000, 0.0
    try:
        balances = api.getbalance()
        if not isinstance(balances, list):
            print(f"❌ 資産状況の取得に失敗しました。APIからの応答: {balances}")
            print("   APIキーが正しいか、Bitflyerのステータスを確認してください。")
            return None, None

        jpy_balance = 0
        eth_balance = 0
        for balance in balances:
            if balance['currency_code'] == 'JPY':
                jpy_balance = balance['available']
            elif balance['currency_code'] == 'ETH':
                eth_balance = balance['available']
        
        # --- ユーザー要望: 1.0 ETHを取引対象外とする ---
        # 実際のETH残高から1.0を差し引いた値を、取引判断に利用する残高とする。
        # これにより、元々保有している1.0 ETHは売買されなくなる。
        trade_eth_balance = max(0, eth_balance - 1.0)
        
        print(f"   実際のETH残高: {eth_balance:.8f} ETH")
        if eth_balance >= 1.0:
             print(f"   ↳ 1.0 ETHを差し引いた【取引対象残高】: {trade_eth_balance:.8f} ETH として処理します。")

        return jpy_balance, trade_eth_balance
    except Exception as e:
        print(f"資産状況の取得中に予期せぬエラーが発生しました: {e}")
        return None, None

def get_ticker(api):
    """現在のETH価格を取得する"""
    try:
        public_api = pybitflyer.API() 
        ticker = public_api.ticker(product_code="ETH_JPY")
        return ticker['ltp']
    except Exception as e:
        print(f"価格の取得に失敗しました: {e}")
        return None

def send_market_order(api, side, size):
    """成行注文を送信する"""
    size = round(size, 8)
    print(f"   注文内容: {side} {size} ETH")
    if config.DRY_RUN:
        print("   -> (DRY RUN) 注文は送信されませんでした。")
        return {'status': 'dry_run'}
    
    try:
        order = api.sendchildorder(
            product_code="ETH_JPY",
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

# --- 状態管理用関数 (損切りロジック用) ---
def get_state_file_path():
    filename = "trade_state_dry.json" if config.DRY_RUN else "trade_state.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def load_entry_price():
    """保存された購入価格を読み込む"""
    path = get_state_file_path()
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get('entry_price')
        except Exception as e:
            print(f"   ⚠️ 状態ファイルの読み込みに失敗しました: {e}")
            return None
    return None

def save_entry_price(price):
    """購入価格を保存する"""
    path = get_state_file_path()
    try:
        with open(path, 'w') as f:
            json.dump({'entry_price': price, 'timestamp': datetime.datetime.now().isoformat()}, f)
        print(f"   💾 取得価格 {price:,.0f} JPY を保存しました。")
    except Exception as e:
        print(f"   ⚠️ 状態ファイルの保存に失敗しました: {e}")

def clear_entry_price():
    """購入価格情報を削除する"""
    path = get_state_file_path()
    if os.path.exists(path):
        try:
            os.remove(path)
            print("   🗑️ 取得価格情報をリセットしました。")
        except Exception as e:
            print(f"   ⚠️ 状態ファイルの削除に失敗しました: {e}")

def run_trading_logic():
    """実際の取引ロジックを実行する"""
    header = "🤖 ETH自動取引ボット実行中 (DRY RUN)" if config.DRY_RUN else "🤖 ETH自動取引ボット実行中 (本番)"
    print("\n" + "="*50)
    print(f"{header} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*50)

    api = get_api()
    try:
        model, scaler = load_checkpoint()
    except FileNotFoundError:
        print("❌ エラー: モデルファイルが見つかりません。")
        print("💡 解決方法: modeling/btc_train.py などを実行してモデルを学習してください。")
        return

    # --- 初期状態確認 ---
    jpy, eth = get_balance(api) # eth変数には1.0を差し引いた値が入る
    if jpy is None:
        return
    print(f"   現在のJPY資産: {jpy:,.0f} JPY")

    # --- データ準備 ---
    # get_btc_dataはETHデータを取得するように変更されている前提
    df = get_btc_data(period="5d", interval="1h") 
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
    print(f"   現在のETH価格: {current_price:,.0f} JPY")

    # 取引対象のETHを保有しているか判断
    has_eth = eth >= MIN_ORDER_ETH 

    if not has_eth:
        if result['class'] == 'up' and result['confidence'] >= config.CONFIDENCE_THRESHOLD:
            print("\n📈 [判断] 購入条件を満たしました。")
            buy_size = (jpy * (1 - config.FEE_RATE)) / current_price
            
            if buy_size >= MIN_ORDER_ETH:
                order = send_market_order(api, "BUY", buy_size)
                if order:
                    # 注文成功時、現在価格を取得価格として保存
                    save_entry_price(current_price)
            else:
                print(f"   -> 購入可能数量 ({buy_size:.8f} ETH) が最小注文数量 ({MIN_ORDER_ETH} ETH) 未満のため、購入を見送ります。")
        else:
            print("\n🧘 [判断] 購入条件を満たさなかったため、待機します。")
    else: # 取引対象のETHを保有している場合
        # --- 損切りチェック ---
        entry_price = load_entry_price()
        is_stop_loss = False
        
        if entry_price:
            stop_loss_price = entry_price * (1 - config.STOP_LOSS_THRESHOLD)
            print(f"   ℹ️ 取得価格: {entry_price:,.0f} JPY | 損切りライン: {stop_loss_price:,.0f} JPY")
            
            if current_price < stop_loss_price:
                print(f"   ⚠️ 現在価格が損切りラインを下回っています！")
                is_stop_loss = True
        else:
            print("   ℹ️ 取得価格情報がありません（手動購入またはファイル紛失）。損切り判定はスキップされます。")

        if result['class'] == 'not-Up' or is_stop_loss:
            if is_stop_loss:
                print("\n📉 [判断] 損切り条件を満たしました。")
            else:
                print("\n📉 [判断] 売却条件を満たしました。")
            
            # 売却するのは取引対象のETHのみ
            sell_size = eth
            
            if sell_size >= MIN_ORDER_ETH:
                order = send_market_order(api, "SELL", sell_size)
                if order:
                    # 売却成功時、取得価格情報をリセット
                    clear_entry_price()
            else:
                print(f"   -> 売却可能数量 ({sell_size:.8f} ETH) が最小注文数量 ({MIN_ORDER_ETH} ETH) 未満のため、売却を見送ります。")
        else:
            print("\n🧘 [判断] 売却条件を満たさなかったため、待機します。")


def main():
    """メインループ"""
    run_trading_logic()

    while True:
        try:
            print(f"\n🕒 次の実行まで {config.HOLD_PERIOD} 時間待機します...")
            time.sleep(config.HOLD_PERIOD * 60 * 60)
            run_trading_logic()
        except FileNotFoundError as e:
            print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py などを実行してモデルを学習してください。")
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
