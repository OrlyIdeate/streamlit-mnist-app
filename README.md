# Streamlit MNIST App

このプロジェクトは、手書きの数字を予測するためのStreamlitアプリケーションです。ユーザーが手書きの数字を入力すると、訓練済みのCNNモデルを使用してその数字を予測します。

## プロジェクト構成

```
streamlit-mnist-app
├── src
│   ├── app.py                # アプリケーションのエントリーポイント
│   └── model
│       └── trained_mnist_cnn_model.pth  # 訓練済みのCNNモデル
├── requirements.txt          # プロジェクトの依存関係
└── README.md                 # プロジェクトのドキュメント
```

## セットアップ手順

1. リポジトリをクローンします。
   ```
   git clone <repository-url>
   cd streamlit-mnist-app
   ```

2. 必要なライブラリをインストールします。
   ```
   pip install -r requirements.txt
   ```

3. アプリケーションを起動します。
   ```
   streamlit run src/app.py
   ```

## 使用方法

アプリケーションが起動したら、手書きの数字を描くためのキャンバスが表示されます。数字を描いた後、「予測」ボタンをクリックすると、モデルが数字を予測します。予測結果が画面に表示されます。

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。