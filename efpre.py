#EFpre
import os
import wget
import torch
import torchvision
import numpy as np
import echonet

def main():
    # Set up directories
    destinationFolder = "destinationFolder"
    videosFolder = "videos_seg/videos"
    os.makedirs(destinationFolder, exist_ok=True)
    DestinationForWeights = "weights"

    # Download weights if not present
    segmentationWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt'
    ejectionFractionWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt'

    # Function to download weights
    #def download_weights(url, destination):
    #    if not os.path.exists(os.path.join(destination, os.path.basename(url))):
    #        st.write(f"Downloading weights from {url} to {destination}")
    #       wget.download(url, out=destination)
    #    else:
    #        st.write("Weights already present")

    # Download Segmentation and EF Weights
    #download_weights(segmentationWeightsURL, DestinationForWeights)
    #download_weights(ejectionFractionWeightsURL, DestinationForWeights)

    # Initialize and Run EF model
    frames = 32
    period = 2
    batch_size = 1  # already set to 1

    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    print("loading weights from ", os.path.join(DestinationForWeights, "r2plus1d_18_32_2_pretrained"))

    if torch.cuda.is_available():
        print("cuda is available, original weights")
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
        model.to(device)
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(ejectionFractionWeightsURL)))
        model.load_state_dict(checkpoint['state_dict'])
        torch.cuda.empty_cache()  # Empty cache after loading model
    else:
        print("cuda is not available, cpu weights")
        device = torch.device("cpu")
        checkpoint = torch.load(os.path.join(DestinationForWeights, os.path.basename(ejectionFractionWeightsURL)), map_location = "cpu")
        state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict_cpu)

    output = os.path.join(destinationFolder, "ef_output.csv")
    
    def filter_filenames(filenames):
        return [f for f in filenames if not f.endswith(".DS_Store")]
    ds = echonet.datasets.Echo(split = "external_test", external_test_location = videosFolder)
    ds.fnames = filter_filenames(ds.fnames)
    print(ds.split, ds.fnames)
    
    mean, std = echonet.utils.get_mean_and_std(ds)

    kwargs = {
        "target_type": "EF",
        "mean": mean,
        "std": std,
        "length": frames,
        "period": period,
        "clips": 12,
    }
    #def custom_collate(batch):
    # テンソルのサイズが [3, 32, 224, 224] と一致するものだけを保持
    #    batch = [item for item in batch if item[0].shape == torch.Size([3, 32, 224, 224])]
        
    #    if not batch:
    #        return torch.Tensor(), torch.Tensor(), torch.Tensor()  
    #    return torch.utils.data.dataloader.default_collate(batch)
    #def filter_filenames(filenames):
    #    return [f for f in filenames if not f.endswith(".DS_Store")]
    ds = echonet.datasets.Echo(split = "external_test", external_test_location = videosFolder, **kwargs)
    ds.fnames = filter_filenames(ds.fnames)
    print(ds.split, ds.fnames)

    test_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False, drop_last=True)

    # Turn off gradient computation for saving memory
    with torch.no_grad():
       # results = echonet.utils.video.run_epoch(model, test_dataloader, False, None, device, save_all=True)
        #print(results)  # 戻り値を確認
        try:
            # run_epoch 関数の戻り値を3つの変数で受け取る
            loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, False, None, device, save_all=True)
            # 以降の処理
        except RuntimeError as e:
            print(f"Error during model inference: {e}")
            return

    # Calculate average EF prediction for each video
    average_predictions = {}
    for (filename, pred) in zip(ds.fnames, yhat):
        if filename not in average_predictions:
            average_predictions[filename] = []
        average_predictions[filename].extend(pred)

    for filename in average_predictions:
        average_predictions[filename] = np.mean(average_predictions[filename])
    return average_predictions

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    avg_predictions = main()  # main関数を呼び出して、平均EF予測値を取得
    # 予測結果を表示
    for filename, avg_pred in avg_predictions.items():
        print(f"{filename}: {avg_pred:.4f}")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()

