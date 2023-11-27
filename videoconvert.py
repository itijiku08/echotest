import os
import shutil
import ffmpeg

def convert_videos(input_folder, output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 入力フォルダ内の全ファイルを処理
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        file_name, file_extension = os.path.splitext(file)

        # 対応する動画ファイル形式のみを変換または移動
        if file_extension.lower() in ['.avi', '.mp4', '.mov', '.mkv']:
            if file_extension.lower() == '.avi':
                # aviファイルは出力フォルダに移動
                shutil.move(file_path, os.path.join(output_folder, file))
            else:
                # 他の形式のファイルはaviに変換して出力フォルダに保存
                output_video_path = os.path.join(output_folder, file_name + ".avi")
                try:
                    ffmpeg.input(file_path).output(output_video_path, vcodec='libxvid').run()
                    print(f'変換完了: {output_video_path}')
                except ffmpeg.Error as e:
                    print('変換に失敗しました。')

if __name__ == "__main__":
    convert_videos('movie', 'videos')
