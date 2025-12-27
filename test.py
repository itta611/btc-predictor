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
    print(
        f"   (学習目標: {config.THR * 100:.2f}%以上の上昇 | 評価基準: {config.EVAL_RETURN_THRESHOLD * 100:.2f}%以上の上昇)")

    df = get_btc_data(period=config.DATA_PERIOD, interval=config.DATA_INTERVAL)
    df_with_features = create_features(df)
    _, valid_mask = create_labels(df_with_features, horizon=config.H, threshold=config.THR)
    df_valid = df_with_features[valid_mask]
    n_total = len(df_valid)
    n_train = int(config.TRAIN_SIZE * n_total)
    n_val = int(config.VAL_SIZE * n_total)
    test_start_index = n_train + n_val

    features = df_valid[config.FEATURE_COLUMNS].values
    prices = df_valid['Close'].values
    y_predictions, y_true_for_eval = [], []

    for i in range(test_start_index + config.L, len(features) - config.H):
        features_seq = features[i - config.L:i]
        result = predict_class(model, scaler, features_seq)
        y_predictions.append(1 if result['class'] == 'up' else 0)

        actual_return = (prices[i + config.H] - prices[i]) / prices[i]
        y_true_for_eval.append(1 if actual_return > config.EVAL_RETURN_THRESHOLD else 0)

    if not y_predictions:
        print("   評価対象データなし。")
        return

    print("\n" + "=" * 50)
    print("📈 評価サマリー")
    print("=" * 50)
    report = classification_report(y_true_for_eval, y_predictions, target_names=config.CLASS_NAMES, output_dict=True,
                                   zero_division=0)
    up_precision = report['Up']['precision']
    print(f"\n🎯 **『上がる』と予測した時の成功率 (適合率): {up_precision:.1%}**")
    print(f"   (「Up」と予測した {report['Up']['support']} 件のうち、実際に価格が上昇した割合)")
    print("\n📊 詳細分類レポート:")
    print(classification_report(y_true_for_eval, y_predictions, target_names=config.CLASS_NAMES, zero_division=0))


def run_trading_simulation(model, scaler, title, offset_days=0):
    """
    指定された期間で取引シミュレーションを実行する。
    ルール：'Up'予測で買い、一定時間後または損切り条件で決済。
    """
    # --- データ準備 ---
    SIM_HOURS = config.SIM_DAYS * 24
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
    exit_time = -1
    entry_price = 0  # 購入価格を記録
    trade_count = 0
    win_count = 0
    stop_loss_count = 0
    portfolio_history = []
    peak_portfolio = initial_balance  # 最高資産の記録
    consecutive_losses = 0  # 連続損失回数
    trade_pause_until = -1  # 取引停止期間

    # --- シミュレーションループ ---
    for i in range(sim_start_index, sim_end_index):
        current_price = prices[i]
        sell = False
        buy = False

        if position == 'long':
            # 1a. 利確決済
            if current_price > entry_price * (1 + config.TAKE_PROFIT_THRESHOLD):
                sell = True

            # # 1b. 損切り決済
            if current_price < entry_price * (1 - config.STOP_LOSS_THRESHOLD):
                sell = True
                stop_loss_count += 1

            # 1c. 時間経過による決済
            if i == exit_time:
                sell = True

            if not sell: continue

        should_buy = False

        if i >= config.L:
            features_seq = features[i - config.L:i]
            result = predict_class(model, scaler, features_seq)
            should_buy = result['class'] == 'up' and result['confidence'] >= config.CONFIDENCE_THRESHOLD

        if position == 'none':
            if should_buy:
                buy = True

        if sell:
            if not should_buy:
                position = 'none'
                # 決済処理
                balance += (btc_amount * current_price) * (1 - config.FEE_RATE)

                # 勝敗判定と連続損失管理
                if current_price * 0.996 > entry_price:
                    win_count += 1
                    consecutive_losses = 0  # 勝利時は連続損失をリセット
                else:
                    consecutive_losses += 1
                    if consecutive_losses >= 3:
                        trade_pause_until = i + 12  # 3連続損失後は12時間取引停止

                btc_amount = 0.0
                trade_count += 1
                print(balance)
            else:
                exit_time = i + config.HOLD_PERIOD
                entry_price = current_price
        # ドローダウンリスク管理
        current_portfolio = balance + (btc_amount * current_price)
        if current_portfolio > peak_portfolio:
            peak_portfolio = current_portfolio
        drawdown = (peak_portfolio - current_portfolio) / peak_portfolio

        # 最低資産比率チェック
        portfolio_ratio = current_portfolio / initial_balance

        if buy and i > trade_pause_until:
            # リスク管理: ドローダウンが大きい時や連続損失時はポジションサイズを減らす
            if drawdown > config.MAX_DRAWDOWN_THRESHOLD or consecutive_losses >= 3:
                position_multiplier = 0.7
            elif portfolio_ratio < config.MIN_PORTFOLIO_RATIO:
                position_multiplier = 0.6
            else:
                position_multiplier = 0.8

            position_size = balance * position_multiplier
            btc_amount = (position_size / current_price) * (1 - config.FEE_RATE)
            balance = balance - position_size
            position = 'long'
            exit_time = i + config.HOLD_PERIOD
            entry_price = current_price

        # ポートフォリオ評価 (毎時間)
        portfolio_value = balance + (btc_amount * current_price)
        portfolio_history.append(portfolio_value)

    # --- 結果集計 ---
    final_portfolio_value = portfolio_history[-1]
    total_return = (final_portfolio_value / initial_balance - 1) * 100
    win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0

    buy_hold_value = (initial_balance / prices[sim_start_index]) * prices[sim_end_index - 1]
    buy_hold_return = (buy_hold_value / initial_balance - 1) * 100

    print(f"\n--- {title} 結果 ---")
    print(f"   最終資産: ${final_portfolio_value:,.2f}")
    print(f"   総リターン: {total_return:.2f}%")
    print(f"   取引回数: {trade_count}回 (勝率: {win_rate:.1f}%)")
    print(f"   損切り回数: {stop_loss_count}回")
    print("--- 比較: Buy & Hold ---")
    print(f"   最終資産: ${buy_hold_value:,.2f}")
    print(f"   総リターン: {buy_hold_return:.2f}%")


def main():
    """メイン関数: モデルの評価と複数期間でのシミュレーションを実行"""
    print("🧪 ビットコイン分類モデル 評価実行")
    print("=" * 60)
    try:
        model, scaler = load_checkpoint()

        # run_evaluation(model, scaler) # 評価は時間がかかるため、一旦コメントアウト

        run_trading_simulation(model, scaler, title="直近30日間", offset_days=100)
        # run_trading_simulation(model, scaler, title="2ヶ月前の30日間", offset_days=360)
    except FileNotFoundError as e:
        print(f"❌ エラー: {e}\n💡 解決方法: modeling/btc_train.py を実行してモデルを学習してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
