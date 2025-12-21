import sys
import os
from trading.bot import TradingBot

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    bot = TradingBot()
    bot.run()

if __name__ == "__main__":
    main()
