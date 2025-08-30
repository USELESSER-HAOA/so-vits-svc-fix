# SoftVC VITS Singing Voice Conversion Fix
[**English**](README.md) | [**中文简体**](README_zh_CN.md)

### I'm sorry, but the Chinese version of the README file is not finished yet.

[![Licence](https://img.shields.io/badge/LICENSE-AGPL3.0-green.svg?style=for-the-badge)](https://github.com/svc-develop-team/so-vits-svc/blob/4.1-Stable/LICENSE)

# Fixed:
1. Supported NVIDIA 50-series GPUs.
2. Badly support to Gloo on the Windows.
3. Upgraded the Python version to 3.10.6.

# How to use

## 1.Follow the README_old.md in the project to get PreTrain model.
## 2.Setup Runtime:
1. Run the Follow command to set up a venv:
```shell
python -m venv .venv
```
2. Enable the venv:
```shell
# Linux
source .venv/bin/activate
# Windows
./.venv/Scripts/activate
```
3. Install dependencies automatically:
```shell
python -m pip install -r requirements.txt
```
- Setup a torch env (Only for 50-series GPUs) :
```shell
# For 50-series GPUs
pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
## 3.Start train
#### The following section is from the original README file.
### Preprocessing

#### 0. Slice audio

To avoid video memory overflow during training or pre-processing, it is recommended to limit the length of audio clips. Cutting the audio to a length of "5s - 15s" is more recommended. Slightly longer times are acceptable, however, excessively long clips may cause problems such as `torch.cuda.OutOfMemoryError`.

To facilitate the slicing process, you can use [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer) or [audio-slicer-CLI](https://github.com/openvpi/audio-slicer)

In general, only the `Minimum Interval` needs to be adjusted. For spoken audio, the default value usually suffices, while for singing audio, it can be adjusted to around `100` or even `50`, depending on the specific requirements.

After slicing, it is recommended to remove any audio clips that are excessively long or too short.

**If you are using whisper-ppg encoder for training, the audio clips must shorter than 30s.**

#### 1. Resample to 44100Hz and mono

```shell
python resample.py
```

##### Cautions

Although this project has resample.py scripts for resampling, mono and loudness matching, the default loudness matching is to match to 0db. This can cause damage to the sound quality. While python's loudness matching package pyloudnorm does not limit the level, this can lead to sonic boom. Therefore, it is recommended to consider using professional sound processing software, such as `adobe audition` for loudness matching. If you are already using other software for loudness matching, add the parameter `-skip_loudnorm` to the run command:

```shell
python resample.py --skip_loudnorm
```

#### 2. Automatically split the dataset into training and validation sets, and generate configuration files.

```shell
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder has the following options

```
vec768l12
vec256l9
hubertsoft
whisper-ppg
cnhubertlarge
dphubert
whisper-ppg-large
wavlmbase+
```

If the speech_encoder argument is omitted, the default value is `vec768l12`

**Use loudness embedding**

Add `--vol_aug` if you want to enable loudness embedding:

```shell
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

After enabling loudness embedding, the trained model will match the loudness of the input source; otherwise, it will match the loudness of the training set.

##### You can modify some parameters in the generated config.json and diffusion.yaml

* `keep_ckpts`: Keep the the the number of previous models during training. Set to `0` to keep them all. Default is `3`.

* `all_in_mem`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.
  
* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the GPU memory capacity.

* `vocoder_name`: Select a vocoder. The default is `nsf-hifigan`.

###### diffusion.yaml

* `cache_all_data`: Load all dataset to RAM. It can be enabled when the disk IO of some platforms is too low and the system memory is **much larger** than your dataset.

* `duration`: The duration of the audio slicing during training, can be adjusted according to the size of the video memory, **Note: this value must be less than the minimum time of the audio in the training set!**

* `batch_size`: The amount of data loaded to the GPU for a single training session can be adjusted to a size lower than the video memory capacity.

* `timesteps`: The total number of steps in the diffusion model, which defaults to 1000.

* `k_step_max`: Training can only train `k_step_max` step diffusion to save training time, note that the value must be less than `timesteps`, 0 is to train the entire diffusion model, **Note: if you do not train the entire diffusion model will not be able to use only_diffusion!**

###### **List of Vocoders**

```
nsf-hifigan
nsf-snake-hifigan
```

#### 3. Generate hubert and f0

```shell
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor has the following options

```
crepe
dio
pm
harvest
rmvpe
fcpe
```

If the training set is too noisy,it is recommended to use `crepe` to handle f0

If the f0_predictor parameter is omitted, the default value is `rmvpe`

If you want shallow diffusion (optional), you need to add the `--use_diff` parameter, for example:

```shell
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

**Speed Up preprocess**

If your dataset is pretty large,you can increase the param `--num_processes` like that:

```shell
python preprocess_hubert_f0.py --f0_predictor dio --num_processes 8
```
All the worker will be assigned to different GPU if you have more than one GPUs.

After completing the above steps, the dataset directory will contain the preprocessed data, and the dataset_raw folder can be deleted.

### Training

#### Sovits Model

```shell
python train.py -c configs/config.json -m 44k
```

#### Diffusion Model (optional)

If the shallow diffusion function is needed, the diffusion model needs to be trained. The diffusion model training method is as follows:

```shell
python train_diff.py -c configs/diffusion.yaml
```

During training, the model files will be saved to `logs/44k`, and the diffusion model will be saved to `logs/44k/diffusion`

# Future Plan
1. Support Colab.
2. Fix this inhumane WebUI.
3. Add more precise control over model training.
4. Support AMD GPUs (No way, Intel GPUs will never be supported under any circumstances.Also,AMD GPUs may could only be used under wsl or pure Linux env) .