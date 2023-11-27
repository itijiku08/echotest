import streamlit as st
import subprocess
import os
import pandas as pd
import efpre

import requests
import os

def download_file(url, destination_folder, filename):
    response = requests.get(url, allow_redirects=True)
    file_path = os.path.join(destination_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {file_path}")

# Dropboxの共有リンクとファイル名のリスト
files_to_download = [
    ("https://www.dropbox.com/scl/fi/4upw2exk2cpzwm3sclabr/deeplabv3_resnet50_random.pt?rlkey=mjj4ab566em6bteevnu6lc2j4&dl=1", "deeplabv3_resnet50_random.pt"),
    ("https://www.dropbox.com/scl/fi/tnvshjd0f4i3sx4p30h2b/r2plus1d_18_32_2_pretrained.pt?rlkey=jr3z6rh70idflfii1ob56scrx&dl=1", "r2plus1d_18_32_2_pretrained.pt")
]

# ダウンロード先のフォルダ
download_folder = "weights"

# フォルダが存在しない場合は作成
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# ファイルをダウンロード
for file_url, filename in files_to_download:
    download_file(file_url, download_folder, filename)



# Streamlitのウェブページ設定
st.title('LVEF計算ソフト')

# ファイルアップロード
uploaded_file = st.file_uploader("動画をアップロードしてください", type=["avi", "mov", "mp4"])

if uploaded_file is not None:
    # videoファイルを保存
    videos_folder_path = "movie"
    if not os.path.exists(videos_folder_path):
        os.makedirs(videos_folder_path)
    saved_file_path = os.path.join(videos_folder_path, uploaded_file.name)
    with open(saved_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 進捗バーの設定
    progress_bar = st.progress(0)
    
    # 先生の予測入力
    doctor_prediction = st.number_input("先生が予測するLVEFを入力してください", min_value=0.0, max_value=100.0, step=0.1)

    progress_bar.progress(10)
    subprocess.run(["python", "videoconvert.py", saved_file_path])
    
    # 動画の分割
    progress_bar.progress(50)
    subprocess.run(["python", "segment.py"])

    # LVEFの予測
    progress_bar.progress(70)
    subprocess.run(["python", "efpre.py"])

    # 予測結果の表示
    progress_bar.progress(90)
    average_predictions = efpre.main()  # efpre.pyから予測値を取得

    progress_bar.progress(100)

    # 予測結果の表示
    for filename, avg_pred in average_predictions.items():
        st.write(f"{filename}: {avg_pred:.4f}")

    # 結果の保存
    if st.button('結果を保存'):
        results = pd.DataFrame({
            "Filename": list(average_predictions.keys()),
            "Predicted EF": list(average_predictions.values()),
            "Doctor's EF": [doctor_prediction] * len(average_predictions)
        })
        results.to_csv("ef_predictions_comparison.csv", index=False)
        st.success("結果が保存されました。")
