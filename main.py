import streamlit as st
import subprocess
import os
import pandas as pd
import efpre


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

    progress_bar.progress(20)
    subprocess.run(["python", "weightdown.py"])

    # 先生の予測入力
    doctor_prediction = st.number_input("先生が予測するLVEFを入力してください", min_value=0.0, max_value=100.0, step=0.1)

    progress_bar.progress(30)
    subprocess.run(["python", "videoconvert.py", saved_file_path])
    
    # 動画の分割
    progress_bar.progress(40)
    subprocess.run(["python", "segment.py"])

    # LVEFの予測
    progress_bar.progress(50)
    subprocess.run(["python", "efpre.py"])

    # 予測結果の表示
    progress_bar.progress(80)
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
