import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# モデルの定義
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"device: {device}")

# モデルのロード
model = CNN()
model.load_state_dict(torch.load('/mount/src/streamlit-mnist-app/src/model/trained_mnist_cnn_model.pth', map_location=torch.device(device)))
model.eval()

# 画像の前処理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # グレースケール変換
        transforms.Resize((28, 28)),  # 28×28にリサイズ
        transforms.ToTensor(),  # テンソル化（値を0-1に正規化）
        transforms.Normalize((0.5,), (0.5,))  # 平均0.5, 標準偏差0.5で正規化
    ])
    return transform(image).unsqueeze(0)  # (1, 1, 28, 28) にリシェイプ

# 予測関数
def predict(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

# Streamlit アプリケーションの設定
st.title("手書き数字認識アプリ")
st.write("手書きの数字を描いて、予測を行います。")

# キャンバスの設定
canvas_result = st_canvas(
    fill_color="black",  # 背景を黒にする
    stroke_color="white",  # ペンの色を白にする
    stroke_width=20,  # 太めに描画（モデルが認識しやすい）
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 画像のアップロードまたは描画
if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert("L")

    # 画像の前処理
    processed_image = preprocess_image(image)

    # 前処理後の画像を表示
    # st.image(processed_image.squeeze().numpy(), caption='前処理後の画像', use_container_width=True, clamp=True)

    # 予測
    prediction = predict(processed_image)
    st.write(f"予測された数字: {prediction}")
