import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.btc_data import get_btc_data, create_features, create_labels
from predictor import predict_class, load_checkpoint
import config

def run_evaluation(model, scaler):
    """
    テストデータ全体でモデルの予測性能を評価する。
    """
    print("\n📊 モデル予測性能評価...")
    print(f"   (学習目標: {config.THR*100:.2f}%以上の上昇 | 評価基準: 0%以上の上昇)")

    df = get_btc_data(period=config.DATA_PERIOD, interval=config.DATA_INTERVAL)
    df_with_features = create_features(df)
    _, valid_mask = create_labels(df_with_features, horizon=config.H, threshold=config.THR)
    df_valid = df_with_features[valid_mask]
    n_total, n_train, n_val = len(df_valid), int(0.7 * len(df_valid)), int(0.15 * len(df_valid))
    test_start_index = n_train + n_val

    features = df_valid[config.FEATURE_COLUMNS].values
    prices = df_valid['Close'].values
    y_predictions, y_true_for_eval = [], []

    for i in range(test_start_index + config.L, len(features) - config.H):
        features_seq = features[i-config.L:i]
        result = predict_class(model, scaler, features_seq)
        y_predictions.append(1 if result['class'] == 'up' else 0)
        
        actual_return = (prices[i + config.H] - prices[i]) / prices[i]
        y_true_for_eval.append(1 if actual_return > 0 else 0)

    if not y_predictions:
        print("   評価対象データなし。")
        return

    print("\n" + "="*50)
    print("📈 評価サマリー")
    print("="*50)
    report = classification_report(y_true_for_eval, y_predictions, target_names=['Not-Up', 'Up'], output_dict=True, zero_division=0)
    up_precision = report['Up']['precision']
    print(f"\n🎯 **『上がる』と予測した時の成功率 (適合率): {up_precision:.1%}**")
    print(f"   (「Up」と予測した {report['Up']['support']} 件のうち、実際に価格が上昇した割合)")
    print("\n📊 詳細分類レポート:")
    print(classification_report(y_true_for_eval, y_predictions, target_names=['Not-Up', 'Up'], zero_division=0))

def run_trading_simulation(model, scaler, title, offset_days=0):
    """
    指定された期間で取引シミュレーションを実行する。
    ルール：'Up'予測で買い、8時間後に強制決済。
    """
    print("\n" + "="*50)
    print(f"📈 取引シミュレーション ({title})")
    print("="*50)

    # --- データ準備 ---
    SIM_DAYS = 30
    SIM_HOURS = SIM_DAYS * 24
    OFFSET_HOURS = offset_days * 24
    
    df = get_btc_data(period=config.DATA_PERIOD, interval="1h")
    df_with_features = create_features(df)
    
    features = df_with_features[config.FEATURE_COLUMNS].values
    prices = df_with_features['Close'].values
    
    sim_end_index = len(features) - OFFSET_HOURS
    sim_start_index = sim_end_index - SIM_HOURS

    if sim_start_index < config.L:
        print(f"   データが不足しているため「{title}」のシミュレーションは実行できません。")
        return

    # --- シミュレーション初期設定 ---
    initial_balance = 10000.0
    balance = initial_balance
    btc_amount = 0.0
    position = 'none'
    exit_time = -1  # ポジションを決済する時刻を保持
    HOLD_PERIOD = 8 # 8時間ホールド
    fee_rate = 0.0004
    confidence_threshold = 0.60
    trade_count = 0
    portfolio_history = []

    # --- シミュレーションループ ---
    for i in range(sim_start_index, sim_end_index):
        current_price = prices[i]
        
        # --- 1. 強制決済の確認 ---
        if position == 'long' and i == exit_time:
            balance = (btc_amount * current_price) * (1 - fee_rate)
            btc_amount = 0.0
            position = 'none'
            exit_time = -1
            trade_count += 1 # 1回のラウンドトリップ取引が完了
            print(f"   {df_with_features.index[i]}: 🔒 SELL (8H Hold) @ ${current_price:,.2f} | Balance: ${balance:,.2f}")
            
            # この時間は決済のみで、新規購入は次の時間から
            portfolio_history.append(balance)
            continue

        # --- 2. 新規購入の判断 (ポジションがない場合のみ) ---
        if position == 'none':
            features_seq = features[i-config.L:i]
            result = predict_class(model, scaler, features_seq)

            if result['class'] == 'up' and result['confidence'] >= confidence_threshold:
                btc_amount = (balance / current_price) * (1 - fee_rate)
                balance = 0.0
                position = 'long'
                exit_time = i + HOLD_PERIOD # 8時間後に売る時間をセット
                print(f"   {df_with_features.index[i]}: 🟢 BUY  @ ${current_price:,.2f}")

        # ポートフォリオ評価 (毎時間)
        portfolio_value = balance + (btc_amount * current_price)
        portfolio_history.append(portfolio_value)

    # --- 結果集計 ---
    final_portfolio_value = portfolio_history[-1]
    total_return = (final_portfolio_value / initial_balance - 1) * 100
    
    buy_hold_value = (initial_balance / prices[sim_start_index]) * prices[sim_end_index-1]
    buy_hold_return = (buy_hold_value / initial_balance - 1) * 100

    print(f"\n--- {title} 結果 ---")
    print(f"   最終資産: ${final_portfolio_value:,.2f}")
    print(f"   総リターン: {total_return:.2f}%")
    print(f"   取引回数: {trade_count}回")
    print("--- 比較: Buy & Hold ---")
    print(f"   最終資産: ${buy_hold_value:,.2f}")
    print(f"   総リターン: {buy_hold_return:.2f}%")


def main():
    """メイン関数: モデルの評価と複数期間でのシミュレーションを実行"""
    print("🧪 ビットコイン分類モデル 評価実行")
    print("=" * 60)
    try:
        model, scaler = load_checkpoint()
        
        run_evaluation(model, scaler)
        
        run_trading_simulation(model, scaler, title="直近30日間", offset_days=0)
        run_trading_simulation(model, scaler, title="2ヶ月前の30日間", offset_days=60)

        print("\n" + "=" * 60)
        print("✅ 全処理完了!")
    except FileNotFoundError as e:
        print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py を実行してモデルを学習してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()