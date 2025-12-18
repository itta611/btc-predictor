# ベースイメージとして公式のPythonイメージを使用
# 軽量なslimバージョンを選択
FROM python:3.11-slim

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 最初に依存関係ファイルのみをコピー
# これにより、ソースコードの変更時にも依存関係のレイヤーはキャッシュされる
COPY requirements.txt .

# pipをアップグレードし、requirements.txtに記載された依存関係をインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# プロジェクトのソースコードをコンテナにコピー
# utilsディレクトリとcheckpointsディレクトリもコピー対象に含める
COPY app.py .
COPY predictor.py .
COPY config.py .
COPY utils/ ./utils/
COPY checkpoints/ ./checkpoints/

# コンテナ起動時に実行するコマンド
# app.pyを実行して取引ボットを起動
CMD ["python", "-u", "app.py"]
