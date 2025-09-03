import gradio as gr
import subprocess
import threading
import queue
import os
import json
import psutil
from datetime import datetime
import sys
from collections import deque
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
import glob

try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf

try:
    import yaml
except ImportError:
    print("Installing PyYAML...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler


class DiffusionLogHandler(FileSystemEventHandler):
    def __init__(self, training_manager):
        self.training_manager = training_manager
        self.monitored_files = {}

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        if any(target in file_path for target in ['saver.log', 'solver.log', 'diffusion.log']):
            self._read_log_file(file_path)

    def _read_log_file(self, file_path):
        try:
            if file_path not in self.monitored_files:
                self.monitored_files[file_path] = 0

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.monitored_files[file_path])
                new_content = f.read()
                self.monitored_files[file_path] = f.tell()

                if new_content.strip():
                    file_name = os.path.basename(file_path)
                    for line in new_content.strip().split('\n'):
                        if line.strip():
                            self.training_manager.add_diffusion_log(f"[{file_name}] {line.strip()}")
        except Exception as e:
            print(f"Error reading log file {file_path}: {e}")


class TrainingManager:
    def __init__(self):
        self.current_process = None
        self.log_queue = queue.Queue()
        self.is_training = False
        self.stop_flag = False
        self.log_history = deque(maxlen=2000)
        self.log_lock = threading.Lock()
        self.current_training_type = None
        self.diffusion_log_observer = None
        self.diffusion_log_handler = None
        self.diffusion_log_paths = [
            "logs/44k/diffusion/",
            "logs/diffusion/",
            "diffusion/logs/",
            "./logs/"
        ]

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        with self.log_lock:
            self.log_history.append(formatted_message)
            self.log_queue.put(formatted_message)

    def add_diffusion_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] 🌊 {message}"
        with self.log_lock:
            self.log_history.append(formatted_message)
            self.log_queue.put(formatted_message)

    def get_all_logs(self):
        with self.log_lock:
            return "\n".join(self.log_history)

    def get_new_logs(self):
        new_logs = []
        while not self.log_queue.empty():
            try:
                new_logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return new_logs

    def clear_logs(self):
        with self.log_lock:
            self.log_history.clear()
            while not self.log_queue.empty():
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    break

    def setup_diffusion_monitoring(self):
        try:
            if self.diffusion_log_observer is None:
                self.diffusion_log_handler = DiffusionLogHandler(self)
                self.diffusion_log_observer = Observer()

                for log_path in self.diffusion_log_paths:
                    if os.path.exists(log_path):
                        self.diffusion_log_observer.schedule(
                            self.diffusion_log_handler,
                            log_path,
                            recursive=True
                        )
                        self.add_log(f"📁 Starting to monitor log directory: {log_path}")

                self.diffusion_log_observer.start()
                self.add_log("🔍 Diffusion model log file monitoring started")
                self.setup_python_log_capture()

        except Exception as e:
            self.add_log(f"Error setting up diffusion monitoring: {str(e)}")

    def setup_python_log_capture(self):
        try:
            class DiffusionLogCaptureHandler(logging.Handler):
                def __init__(self, training_manager):
                    super().__init__()
                    self.training_manager = training_manager

                def emit(self, record):
                    try:
                        if any(target in record.pathname for target in
                               ['diffusion/logger/saver.py', 'diffusion/solver.py', 'saver.py', 'solver.py']):

                            module_name = os.path.basename(record.pathname)
                            log_message = f"[{module_name}:{record.lineno}] {record.getMessage()}"

                            if 'saver.py' in module_name:
                                log_message = f"💾 {log_message}"
                            elif 'solver.py' in module_name:
                                log_message = f"🧮 {log_message}"

                            self.training_manager.add_diffusion_log(log_message)
                    except Exception:
                        pass

            diffusion_handler = DiffusionLogCaptureHandler(self)
            diffusion_handler.setLevel(logging.DEBUG)
            root_logger = logging.getLogger()
            root_logger.addHandler(diffusion_handler)
            root_logger.setLevel(logging.DEBUG)
            self.add_log("🎯 Python log interceptor setup, monitoring saver.py and solver.py")

        except Exception as e:
            self.add_log(f"Error setting up Python log capture: {str(e)}")

    def stop_diffusion_monitoring(self):
        try:
            if self.diffusion_log_observer:
                self.diffusion_log_observer.stop()
                self.diffusion_log_observer.join()
                self.diffusion_log_observer = None
                self.diffusion_log_handler = None
                self.add_log("🔍 Diffusion model log monitoring stopped")
        except Exception as e:
            self.add_log(f"Error stopping diffusion monitoring: {str(e)}")

    def read_output(self, process, log_queue):
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_message = output.strip()

                if self.current_training_type == 'diff':
                    log_message = f"[TRAIN_DIFF] {log_message}"
                elif self.current_training_type == 'main':
                    log_message = f"[MAIN] {log_message}"
                elif self.current_training_type == 'inference':
                    log_message = f"[INFERENCE] {log_message}"

                self.add_log(log_message)

    def run_command(self, cmd, cwd=None):
        try:
            self.add_log(f"Executing command: {cmd}")

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True,
                cwd=cwd,
                env=dict(os.environ, PYTHONUNBUFFERED='1')
            )

            thread = threading.Thread(target=self.read_output, args=(self.current_process, self.log_queue))
            thread.daemon = True
            thread.start()

            self.current_process.wait()

            if self.current_process.returncode == 0:
                self.add_log("Command executed successfully")
                return True
            else:
                self.add_log(f"Command execution failed, exit code: {self.current_process.returncode}")
                return False

        except Exception as e:
            self.add_log(f"Error executing command: {str(e)}")
            return False

    def stop_training(self):
        if self.current_process:
            try:
                self.add_log("Stopping training process...")

                parent = psutil.Process(self.current_process.pid)
                children = parent.children(recursive=True)

                for child in children:
                    child.terminate()

                parent.terminate()
                gone, still_alive = psutil.wait_procs(children + [parent], timeout=5)

                for p in still_alive:
                    p.kill()

                self.add_log("Training stopped")
            except Exception as e:
                self.add_log(f"Error stopping training: {str(e)}")
            finally:
                self.current_process = None
                self.is_training = False
                self.stop_flag = True
                self.current_training_type = None
                if self.current_training_type == 'diff':
                    self.stop_diffusion_monitoring()


training_manager = TrainingManager()


def get_available_models():
    models = []
    model_extensions = ['.pth', '.pt']

    for root, dirs, files in os.walk('./logs'):
        for file in files:
            if any(file.endswith(ext) for ext in model_extensions) and 'G_' in file:
                models.append(os.path.join(root, file))

    return sorted(models) if models else ["No models found"]


def get_available_configs():
    configs = []

    if os.path.exists('./configs/config.json'):
        configs.append('./configs/config.json')

    for root, dirs, files in os.walk('./logs'):
        for file in files:
            if file == 'config.json':
                configs.append(os.path.join(root, file))

    return sorted(configs) if configs else ["No config files found"]


def get_available_diffusion_models():
    models = []
    diffusion_paths = [
        './logs/44k/diffusion',
        './logs/diffusion',
        './diffusion',
    ]

    for path in diffusion_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.pt') and ('model_' in file or 'diffusion_' in file):
                        models.append(os.path.join(root, file))

    return sorted(models) if models else ["No diffusion models found"]


def get_available_diffusion_configs():
    configs = []
    if os.path.exists('./configs/diffusion.yaml'):
        configs.append('./configs/diffusion.yaml')
    return sorted(configs) if configs else ["No diffusion config files found"]


def get_speakers_from_config(config_path):
    try:
        if not config_path or not os.path.exists(config_path):
            return ["No config selected"]

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        speakers = config.get('spk', {}).keys()
        return list(speakers) if speakers else ["No speakers found in config"]
    except Exception as e:
        return [f"Error reading config: {str(e)}"]


def check_dataset_structure(dataset_path):
    if not os.path.exists(dataset_path):
        return False, "Dataset path does not exist"

    speakers = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not speakers:
        return False, "No speaker folders found"

    total_files = 0
    for speaker in speakers:
        speaker_path = os.path.join(dataset_path, speaker)
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        total_files += len(wav_files)

    if total_files == 0:
        return False, "No WAV files found"

    return True, f"Found {len(speakers)} speakers with {total_files} audio files total"


def resample_audio(dataset_raw_path, dataset_44k_path, skip_loudnorm, progress=gr.Progress()):
    training_manager.stop_flag = False
    progress(0, desc="Starting resample...")

    cmd = f"python resample.py --in_dir {dataset_raw_path} --out_dir2 {dataset_44k_path}"
    if skip_loudnorm:
        cmd += " --skip_loudnorm"

    success = training_manager.run_command(cmd)

    if success:
        training_manager.add_log("✅ Resample completed")
        return "✅ Resample completed", training_manager.get_all_logs()
    else:
        training_manager.add_log("❌ Resample failed")
        return "❌ Resample failed", training_manager.get_all_logs()


def preprocess_config(dataset_44k_path, speech_encoder, vol_aug, tiny_model, progress=gr.Progress()):
    training_manager.stop_flag = False
    progress(0, desc="Generating config file...")

    cmd = f"python preprocess_flist_config.py --source_dir {dataset_44k_path} --speech_encoder {speech_encoder}"
    if vol_aug:
        cmd += " --vol_aug"
    if tiny_model:
        cmd += " --tiny"

    success = training_manager.run_command(cmd)

    if success:
        if os.path.exists("configs/config.json"):
            with open("configs/config.json", "r") as f:
                config = json.load(f)
                n_speakers = config["model"]["n_speakers"]
                training_manager.add_log(f"Config file generated: {n_speakers} speakers")
        training_manager.add_log("✅ Config file generation completed")
        return "✅ Config file generation completed", training_manager.get_all_logs()
    else:
        training_manager.add_log("❌ Config file generation failed")
        return "❌ Config file generation failed", training_manager.get_all_logs()


def extract_features(dataset_44k_path, f0_predictor, use_diff, num_processes, device, progress=gr.Progress()):
    training_manager.stop_flag = False
    progress(0, desc="Extracting features...")

    cmd = f"python preprocess_hubert_f0.py --in_dir {dataset_44k_path} --f0_predictor {f0_predictor} --num_processes {num_processes}"

    if use_diff:
        cmd += " --use_diff"
    if device and device != "auto":
        cmd += f" --device {device}"

    success = training_manager.run_command(cmd)

    if success:
        training_manager.add_log("✅ Feature extraction completed")
        return "✅ Feature extraction completed", training_manager.get_all_logs()
    else:
        training_manager.add_log("❌ Feature extraction failed")
        return "❌ Feature extraction failed", training_manager.get_all_logs()


def start_training(model_name, use_ascend, progress=gr.Progress()):
    if training_manager.is_training:
        return "⚠️ Training already in progress", training_manager.get_all_logs()

    training_manager.is_training = True
    training_manager.stop_flag = False
    training_manager.current_training_type = 'main'

    model_dir = f"logs/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    if use_ascend:
        cmd = f"python train_ascend.py -c configs/config.json -m {model_name}"
    else:
        cmd = f"python train.py -c configs/config.json -m {model_name}"

    def train_thread():
        training_manager.add_log(f"🚀 Starting main model training")
        training_manager.add_log(f"Model will be saved to: {model_dir}")
        training_manager.add_log(f"Using device: {'Ascend NPU' if use_ascend else 'NVIDIA GPU'}")

        success = training_manager.run_command(cmd)
        training_manager.is_training = False
        training_manager.current_training_type = None

        if success:
            training_manager.add_log("✅ Main model training completed!")
        else:
            training_manager.add_log("❌ Main model training failed or interrupted")

    thread = threading.Thread(target=train_thread)
    thread.daemon = True
    thread.start()

    return "🚀 Training started", training_manager.get_all_logs()


def start_diff_training(use_ascend, progress=gr.Progress()):
    if training_manager.is_training:
        return "⚠️ Training already in progress", training_manager.get_all_logs()

    training_manager.is_training = True
    training_manager.stop_flag = False
    training_manager.current_training_type = 'diff'

    config_path = "configs/diffusion.yaml"
    if not os.path.exists(config_path):
        training_manager.add_log(f"❌ Config file does not exist: {config_path}")
        training_manager.add_log("Please complete preprocessing steps to generate config file")
        training_manager.is_training = False
        training_manager.current_training_type = None
        return "❌ Config file does not exist", training_manager.get_all_logs()

    if use_ascend:
        cmd = f"python train_diff_ascend.py -c configs/diffusion.yaml"
    else:
        cmd = f"python train_diff.py -c configs/diffusion.yaml"

    def train_thread():
        training_manager.add_log(f"🌊 Starting diffusion model training")
        training_manager.add_log(f"Using device: {'Ascend NPU' if use_ascend else 'NVIDIA GPU'}")
        training_manager.add_log(f"Diffusion model will be saved to: logs/44k/diffusion")

        training_manager.setup_diffusion_monitoring()
        training_manager.add_log(f"🔍 Started capturing output from diffusion/logger/saver.py and diffusion/solver.py")
        training_manager.add_log(f"Starting diffusion model training process...")

        success = training_manager.run_command(cmd)
        training_manager.is_training = False
        training_manager.current_training_type = None

        training_manager.stop_diffusion_monitoring()

        if success:
            training_manager.add_log("✅ Diffusion model training completed!")
        else:
            training_manager.add_log("❌ Diffusion model training failed or interrupted")

    thread = threading.Thread(target=train_thread)
    thread.daemon = True
    thread.start()

    return "🚀 Diffusion model training started", training_manager.get_all_logs()


def run_inference(
        model_path, uploaded_model, config_path, uploaded_config,
        input_audio, uploaded_audio,
        trans, speaker,
        clip_duration, linear_gradient, f0_predictor, auto_predict_f0,
        slice_db, device_selection, noise_scale, pad_seconds, wav_format,
        linear_gradient_retain, enhancer_adaptive_key, f0_filter_threshold,
        use_diffusion, diffusion_model_path, uploaded_diff_model,
        diffusion_config_path, uploaded_diff_config, k_step, only_diffusion, second_encoding,
        enhance, cluster_model_path, cluster_infer_ratio, feature_retrieval,
        use_spk_mix, loudness_envelope_adjustment,
        progress=gr.Progress()
):
    training_manager.current_training_type = 'inference'
    training_manager.add_log("🎤 Starting voice conversion inference...")

    os.makedirs("raw", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    final_model_path = model_path
    if uploaded_model and uploaded_model.name:
        final_model_path = uploaded_model.name
        training_manager.add_log(f"Using uploaded model: {os.path.basename(uploaded_model.name)}")

    final_config_path = config_path
    if uploaded_config and uploaded_config.name:
        final_config_path = uploaded_config.name
        training_manager.add_log(f"Using uploaded config: {os.path.basename(uploaded_config.name)}")

    input_audio_path = None
    if uploaded_audio and uploaded_audio.name:
        audio_filename = os.path.basename(uploaded_audio.name)
        input_audio_path = os.path.join("raw", audio_filename)
        shutil.copy(uploaded_audio.name, input_audio_path)
        training_manager.add_log(f"Using uploaded audio: {audio_filename}")
    elif input_audio:
        audio_filename = "recorded_audio.wav"
        input_audio_path = os.path.join("raw", audio_filename)
        import soundfile as sf
        sf.write(input_audio_path, input_audio[1], input_audio[0])
        training_manager.add_log(f"Using recorded audio saved as: {audio_filename}")

    if not input_audio_path or not os.path.exists(input_audio_path):
        error_msg = "❌ No input audio provided"
        training_manager.add_log(error_msg)
        return None, gr.File(visible=False), error_msg

    if not final_model_path or final_model_path == "No models found" or not os.path.exists(final_model_path):
        error_msg = "❌ Model file not found"
        training_manager.add_log(error_msg)
        return None, gr.File(visible=False), error_msg

    if not final_config_path or final_config_path == "No config files found" or not os.path.exists(final_config_path):
        error_msg = "❌ Config file not found"
        training_manager.add_log(error_msg)
        return None, gr.File(visible=False), error_msg

    audio_name = os.path.splitext(os.path.basename(input_audio_path))[0]

    cmd = f'python inference_main.py -m "{final_model_path}" -c "{final_config_path}" -n "{audio_name}" -t {trans} -s "{speaker}"'

    if clip_duration > 0:
        cmd += f" -cl {clip_duration}"
    if linear_gradient > 0:
        cmd += f" -lg {linear_gradient}"
    if f0_predictor != "pm":
        cmd += f" -f0p {f0_predictor}"
    if auto_predict_f0:
        cmd += " -a"

    cmd += f" -sd {slice_db}"

    if device_selection != "auto":
        cmd += f" -d {device_selection}"

    cmd += f" -ns {noise_scale}"
    cmd += f" -p {pad_seconds}"
    cmd += f" -wf {wav_format}"
    cmd += f" -lgr {linear_gradient_retain}"

    if enhancer_adaptive_key != 0:
        cmd += f" -eak {enhancer_adaptive_key}"

    cmd += f" -ft {f0_filter_threshold}"

    if enhance:
        cmd += " -eh"
    if cluster_model_path and cluster_model_path.strip():
        cmd += f' -cm "{cluster_model_path}"'
        if cluster_infer_ratio > 0:
            cmd += f" -cr {cluster_infer_ratio}"
    if feature_retrieval:
        cmd += " -fr"

    if use_spk_mix:
        cmd += " -usm"
    if loudness_envelope_adjustment < 1.0:
        cmd += f" -lea {loudness_envelope_adjustment}"

    if use_diffusion:
        final_diffusion_model_path = diffusion_model_path
        if uploaded_diff_model and uploaded_diff_model.name:
            final_diffusion_model_path = uploaded_diff_model.name
            training_manager.add_log(f"Using uploaded diffusion model: {os.path.basename(uploaded_diff_model.name)}")

        final_diffusion_config_path = diffusion_config_path
        if uploaded_diff_config and uploaded_diff_config.name:
            final_diffusion_config_path = uploaded_diff_config.name
            training_manager.add_log(f"Using uploaded diffusion config: {os.path.basename(uploaded_diff_config.name)}")

        if final_diffusion_model_path and final_diffusion_model_path != "No diffusion models found" and os.path.exists(
                final_diffusion_model_path):
            cmd += f' -dm "{final_diffusion_model_path}"'

        if final_diffusion_config_path and final_diffusion_config_path != "No diffusion config files found" and os.path.exists(
                final_diffusion_config_path):
            cmd += f' -dc "{final_diffusion_config_path}"'

        cmd += " -shd"
        cmd += f" -ks {k_step}"

        if only_diffusion:
            cmd += " -od"
        if second_encoding:
            cmd += " -se"

    try:
        with open(final_config_path, 'r') as f:
            config = json.load(f)
            speech_encoder = config.get('model', {}).get('speech_encoder', '')
            if 'whisper-ppg' in speech_encoder:
                cmd += " -cl 25 -lg 1"
                training_manager.add_log(
                    "🎵 Detected whisper-ppg encoder, using recommended settings (clip=25, linear_gradient=1)")
    except:
        pass

    training_manager.add_log(f"🎯 Inference parameters configured:")
    training_manager.add_log(f"   Model: {os.path.basename(final_model_path)}")
    training_manager.add_log(f"   Config: {os.path.basename(final_config_path)}")
    training_manager.add_log(f"   Audio: {os.path.basename(input_audio_path)}")
    training_manager.add_log(f"   Speaker: {speaker}")
    training_manager.add_log(f"   Pitch shift: {trans}")
    training_manager.add_log(
        f"   Audio processing: slice_db={slice_db}, noise_scale={noise_scale}, pad_seconds={pad_seconds}")
    training_manager.add_log(f"   Output format: {wav_format}")
    if use_diffusion:
        training_manager.add_log(f"   Using diffusion model with {k_step} steps")

    def inference_thread():
        nonlocal cmd
        success = training_manager.run_command(cmd)
        training_manager.current_training_type = None
        return success

    success = inference_thread()

    if success:
        key = "auto" if auto_predict_f0 else f"{trans}key"
        cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
        isdiffusion = "sovits"
        if use_diffusion:
            if only_diffusion:
                isdiffusion = "diff"
            else:
                isdiffusion = "sovdiff"
        final_speaker = "spk_mix" if use_spk_mix else speaker
        expected_filename = f"{audio_name}_{key}_{final_speaker}{cluster_name}_{isdiffusion}_{f0_predictor}.{wav_format}"
        expected_filepath = os.path.join("results", expected_filename)

        training_manager.add_log(f"🔍 Looking for output file: {expected_filename}")

        if os.path.exists(expected_filepath):
            training_manager.add_log(f"✅ Found output file: {expected_filename}")
            training_manager.add_log(f"📥 File available for download: {expected_filename}")
            return expected_filepath, gr.File(value=expected_filepath,
                                              visible=True), f"✅ Voice conversion completed! Output: {expected_filename}"

        results_dir = "results"
        patterns = [
            f"{audio_name}_*_{final_speaker}*_{isdiffusion}_{f0_predictor}.{wav_format}",
            f"{audio_name}_*_{final_speaker}*_{isdiffusion}_*.{wav_format}",
            f"{audio_name}_*_{final_speaker}*.{wav_format}",
            f"{audio_name}*.{wav_format}"
        ]

        output_files = []
        for pattern in patterns:
            pattern_files = glob.glob(os.path.join(results_dir, pattern))
            if pattern_files:
                output_files.extend(pattern_files)
                training_manager.add_log(
                    f"📁 Found files with pattern '{pattern}': {[os.path.basename(f) for f in pattern_files]}")
                break

        if output_files:
            output_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            output_file = output_files[0]
            training_manager.add_log(f"🎵 Using output file: {os.path.basename(output_file)}")
            training_manager.add_log(f"📥 File available for download: {os.path.basename(output_file)}")
            return output_file, gr.File(value=output_file,
                                        visible=True), f"✅ Voice conversion completed! Output: {os.path.basename(output_file)}"

        try:
            all_files = os.listdir(results_dir)
            training_manager.add_log(f"📂 All files in results directory: {all_files}")
        except:
            pass

        error_msg = f"⚠️ Voice conversion completed but output file not found. Expected: {expected_filename}"
        training_manager.add_log(error_msg)
        return None, gr.File(visible=False), error_msg
    else:
        error_msg = "❌ Voice conversion failed"
        training_manager.add_log(error_msg)
        return None, gr.File(visible=False), error_msg


def stop_training():
    if not training_manager.is_training:
        training_manager.add_log("⚠️ No training currently in progress")
        return "⚠️ No training currently in progress", training_manager.get_all_logs()

    training_manager.stop_training()
    return "⏹️ Stopping training...", training_manager.get_all_logs()


def stop_training_main():
    return stop_training()


def stop_training_diff():
    return stop_training()


def update_logs():
    return training_manager.get_all_logs()


def update_logs_with_scroll():
    logs = training_manager.get_all_logs()
    log_count = len(training_manager.log_history)
    if log_count > 0:
        print(f"[DEBUG] Updating logs: {log_count} total logs")
    return logs, logs, logs, logs


def force_update_diff_logs():
    logs = training_manager.get_all_logs()
    print(f"[DEBUG] Force updating diffusion model logs: {len(logs)} characters")
    return logs


def clear_logs():
    training_manager.clear_logs()
    return ""


def test_diffusion_monitoring():
    training_manager.setup_diffusion_monitoring()
    training_manager.add_log("🧪 Diffusion model monitoring test started")
    training_manager.add_log("📝 Now will capture output from saver.py and solver.py")
    return "🧪 Test monitoring started", training_manager.get_all_logs()


with gr.Blocks(title="So-VITs-SVC-Fix WebUI") as app:
    gr.Markdown("""
    # So-VITs-SVC-Fix WebUI

    ### Usage Flow:
    1. **Data Preparation**: Put audio files into `dataset_raw/speaker_name/` folders
    2. **Preprocessing**: Execute in order: Resample → Generate Config → Extract Features
    3. **Training**: Configure parameters and start training
    4. **Inference**: Use trained models for voice conversion
    5. **Monitoring**: View real-time logs to understand training progress
    """)

    with gr.Tabs():
        with gr.TabItem("📁 Data Preprocessing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Audio Resampling")
                    dataset_raw_path = gr.Textbox(
                        label="Raw Dataset Path",
                        value="./dataset_raw",
                        info="Path containing speaker folders"
                    )
                    dataset_44k_path = gr.Textbox(
                        label="Output Path (44.1kHz)",
                        value="./dataset/44k"
                    )
                    skip_loudnorm = gr.Checkbox(
                        label="Skip Loudness Normalization",
                        value=False,
                        info="Check if loudness has already been processed"
                    )
                    resample_btn = gr.Button("🔄 Start Resampling", variant="primary")
                    resample_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    gr.Markdown("### Step 2: Generate Configuration")
                    speech_encoder = gr.Dropdown(
                        label="Speech Encoder",
                        choices=["vec768l12", "vec256l9", "hubertsoft", "whisper-ppg",
                                 "cnhubertlarge", "dphubert", "whisper-ppg-large", "wavlmbase+"],
                        value="vec768l12"
                    )
                    vol_aug = gr.Checkbox(
                        label="Enable Volume Augmentation",
                        value=False,
                        info="Use volume embedding during training"
                    )
                    tiny_model = gr.Checkbox(
                        label="Use Tiny Model",
                        value=False,
                        info="Use smaller model architecture"
                    )
                    config_btn = gr.Button("📝 Generate Config", variant="primary")
                    config_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 3: Extract Features")
                    f0_predictor = gr.Dropdown(
                        label="F0 Predictor",
                        choices=["rmvpe", "crepe", "dio", "pm", "harvest", "fcpe"],
                        value="rmvpe",
                        info="Recommended to use rmvpe"
                    )
                    use_diff = gr.Checkbox(
                        label="Use Diffusion Model",
                        value=False,
                        info="Extract additional features for diffusion model"
                    )
                    num_processes = gr.Slider(
                        label="Number of Parallel Processes",
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        info="Set according to CPU core count"
                    )
                    device_choice = gr.Dropdown(
                        label="Device Selection",
                        choices=["auto", "cuda:0", "cuda:1", "cpu"],
                        value="auto"
                    )
                    extract_btn = gr.Button("🎯 Extract Features", variant="primary")
                    extract_status = gr.Textbox(label="Status", interactive=False)

                with gr.Column():
                    gr.Markdown("### Dataset Check")
                    check_dataset_btn = gr.Button("🔍 Check Dataset Structure")
                    dataset_info = gr.Textbox(label="Dataset Information", lines=3, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 📊 Preprocessing Logs")
            with gr.Row():
                preprocess_refresh_btn = gr.Button("🔄 Refresh Logs", scale=1)
                preprocess_clear_btn = gr.Button("🗑️ Clear Logs", scale=1, variant="secondary")

            preprocess_log_display = gr.Textbox(
                label="Preprocessing Log Output",
                lines=15,
                max_lines=20,
                interactive=False,
                elem_id="preprocess_log_display",
                info="Shows real-time logs for resampling, config generation, and feature extraction"
            )

        with gr.TabItem("⚙️ Main Model Training"):
            gr.Markdown("""
            ### Main Model Training
            Training command will use: `python train.py -c configs/config.json -m {model_name}`

            Configuration file `configs/config.json` needs to be generated through preprocessing steps, or manually edited to modify training parameters.
            """)

            with gr.Row():
                with gr.Column():
                    model_name = gr.Textbox(
                        label="Model Name",
                        value="44k",
                        info="Model will be saved in logs/{model_name} folder"
                    )
                    use_ascend = gr.Checkbox(
                        label="Use Ascend NPU",
                        value=False,
                        info="Use Huawei Ascend card for training"
                    )

                with gr.Column():
                    gr.Markdown("""
                    #### 📝 Configuration Notes
                    - For batch size, learning rate and other parameters, directly edit `configs/config.json`
                    - keep_ckpts: Number of checkpoints to keep
                    - batch_size: Adjust according to GPU memory
                    - learning_rate: Initial learning rate
                    - epochs: Number of training epochs
                    """)

            with gr.Row():
                train_btn = gr.Button("🚀 Start Training", variant="primary", scale=2)
                stop_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)

            train_status = gr.Textbox(label="Training Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 📊 Training Logs")
            with gr.Row():
                train_refresh_btn = gr.Button("🔄 Refresh Logs", scale=1)
                train_clear_btn = gr.Button("🗑️ Clear Logs", scale=1, variant="secondary")

            train_log_display = gr.Textbox(
                label="Main Model Training Logs",
                lines=20,
                max_lines=25,
                interactive=False,
                elem_id="train_log_display",
                info="Shows real-time logs and progress for main model training"
            )

        with gr.TabItem("🌊 Diffusion Model Training"):
            gr.Markdown("""
            ### Diffusion Model Training (Optional)
            Training command will use: `python train_diff.py -c configs/diffusion.yaml`

            Configuration file `configs/diffusion.yaml` needs to be generated through preprocessing steps, or manually edited to modify parameters.

            **🔍 Advanced Monitoring**: System will directly capture output from the following modules and integrate into training logs:
            - `diffusion/logger/saver.py` - Model save and load operations
            - `diffusion/solver.py` - Training solver core logic
            """)

            with gr.Row():
                with gr.Column():
                    diff_use_ascend = gr.Checkbox(
                        label="Use Ascend NPU",
                        value=False,
                        info="Use Huawei Ascend card for training"
                    )

                    test_monitor_btn = gr.Button("🧪 Test Monitoring", variant="secondary")

                with gr.Column():
                    gr.Markdown("""
                    #### 📝 Diffusion Model Configuration Notes
                    Please edit `configs/diffusion.yaml` to modify the following parameters:
                    - batch_size: Batch size
                    - lr: Learning rate
                    - timesteps: Diffusion steps (default 1000)
                    - k_step_max: Maximum K steps (0 for full model)
                    - cache_all_data: Whether to cache all data in memory

                    #### 🔍 Monitoring Notes
                    Training will automatically capture:
                    - 💾 [saver.py] Model save/load logs
                    - 🧮 [solver.py] Solver runtime logs
                    - 📁 [*.log] Related log file contents
                    """)

            with gr.Row():
                diff_train_btn = gr.Button("🚀 Start Diffusion Model Training", variant="primary", scale=2)
                diff_stop_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)

            diff_train_status = gr.Textbox(label="Training Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 📊 Diffusion Model Training Logs")
            with gr.Row():
                diff_refresh_btn = gr.Button("🔄 Refresh Logs", scale=1)
                diff_clear_btn = gr.Button("🗑️ Clear Logs", scale=1, variant="secondary")

            diff_log_display = gr.Textbox(
                label="Diffusion Model Training Logs (including saver.py and solver.py output)",
                lines=20,
                max_lines=25,
                interactive=False,
                elem_id="diff_log_display",
                info="Shows complete diffusion model training logs, including directly captured saver.py and solver.py output"
            )

        with gr.TabItem("🎤 Voice Conversion Inference"):
            gr.Markdown("""
            ### Voice Conversion Inference
            Use your trained models to convert voices. Supports both main model and diffusion model inference.
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎯 Model Selection")

                    with gr.Group():
                        gr.Markdown("**Main Model**")
                        inf_model_path = gr.Dropdown(
                            label="Main Model Path",
                            choices=get_available_models(),
                            value=get_available_models()[0] if get_available_models()[0] != "No models found" else None,
                            info="Select trained model file",
                            allow_custom_value=True
                        )
                        inf_uploaded_model = gr.File(
                            label="Or Upload Model File (.pth/.pt)",
                            file_types=[".pth", ".pt"]
                        )

                        inf_config_path = gr.Dropdown(
                            label="Config Path",
                            choices=get_available_configs(),
                            value=get_available_configs()[0] if get_available_configs()[
                                                                    0] != "No config files found" else None,
                            info="Select config.json file"
                        )
                        inf_uploaded_config = gr.File(
                            label="Or Upload Config File (.json)",
                            file_types=[".json"]
                        )

                    refresh_models_btn = gr.Button("🔄 Refresh Model Lists", variant="secondary")

                with gr.Column():
                    gr.Markdown("#### 🎵 Audio Input")

                    with gr.Group():
                        inf_input_audio = gr.Audio(
                            label="Record Audio",
                            sources=["microphone"],
                            type="numpy",
                        )
                        inf_uploaded_audio = gr.File(
                            label="Or Upload Audio File (.wav/.mp3/.flac)",
                            file_types=[".wav", ".mp3", ".flac"]
                        )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ⚙️ Basic Settings")

                    inf_trans = gr.Slider(
                        label="Pitch Shift (semitones)",
                        minimum=-24,
                        maximum=24,
                        value=0,
                        step=1,
                        info="Positive for higher pitch, negative for lower pitch"
                    )

                    inf_speaker = gr.Dropdown(
                        label="Target Speaker",
                        choices=get_speakers_from_config(get_available_configs()[0] if get_available_configs()[
                                                                                           0] != "No config files found" else None),
                        info="Select target speaker for conversion"
                    )

                    inf_clip_duration = gr.Slider(
                        label="Clip Duration (seconds)",
                        minimum=0,
                        maximum=60,
                        value=0,
                        step=1,
                        info="0 for automatic slicing, >0 for forced slicing"
                    )

                with gr.Column():
                    gr.Markdown("#### 🔧 Advanced Settings")

                    inf_linear_gradient = gr.Slider(
                        label="Linear Gradient (seconds)",
                        minimum=0,
                        maximum=5,
                        value=0,
                        step=0.1,
                        info="Cross-fade length for audio segments"
                    )

                    inf_f0_predictor = gr.Dropdown(
                        label="F0 Predictor",
                        choices=["pm", "crepe", "dio", "harvest", "rmvpe", "fcpe"],
                        value="rmvpe",
                        info="F0 prediction method"
                    )

                    inf_auto_predict_f0 = gr.Checkbox(
                        label="Auto Predict F0",
                        value=False,
                        info="⚠️ Don't use for singing voice conversion"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🎚️ Audio Processing Settings")

                    inf_slice_db = gr.Slider(
                        label="Audio Slice Threshold (dB)",
                        minimum=-60,
                        maximum=-10,
                        value=-40,
                        step=1,
                        info="Audio slicing level threshold. -30 for noisy audio, -50 to retain breath sounds"
                    )

                    inf_device_selection = gr.Dropdown(
                        label="Inference Device",
                        choices=["auto", "cpu", "cuda:0", "cuda:1"],
                        value="auto",
                        info="Device for inference, auto for automatic selection"
                    )

                    inf_noise_scale = gr.Slider(
                        label="Noise Scale",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.4,
                        step=0.01,
                        info="Affects pronunciation and audio quality, somewhat mystical parameter"
                    )

                    inf_pad_seconds = gr.Slider(
                        label="Padding Seconds",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        info="Add silence padding to avoid artifacts at beginning/end"
                    )

                with gr.Column():
                    gr.Markdown("#### 📁 Output & Processing Settings")

                    inf_wav_format = gr.Dropdown(
                        label="Output Audio Format",
                        choices=["wav", "flac", "mp3"],
                        value="flac",
                        info="Output audio file format"
                    )

                    inf_linear_gradient_retain = gr.Slider(
                        label="Linear Gradient Retain Ratio",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.75,
                        step=0.01,
                        info="Cross-fade length retention ratio for auto-sliced audio"
                    )

                    inf_enhancer_adaptive_key = gr.Slider(
                        label="Enhancer Adaptive Key (semitones)",
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        info="Make enhancer adapt to higher pitch ranges"
                    )

                    inf_f0_filter_threshold = gr.Slider(
                        label="F0 Filter Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.05,
                        step=0.01,
                        info="Only effective when using CREPE. Lower values reduce off-key probability but may increase muted sounds"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🌊 Diffusion Model Settings")

                    inf_use_diffusion = gr.Checkbox(
                        label="Use Diffusion Model",
                        value=False,
                        info="Enable diffusion model for better quality"
                    )

                    with gr.Group():
                        inf_diffusion_model_path = gr.Dropdown(
                            label="Diffusion Model Path",
                            choices=get_available_diffusion_models(),
                            value=get_available_diffusion_models()[0] if get_available_diffusion_models()[
                                                                             0] != "No diffusion models found" else None,
                            info="Select diffusion model file"
                        )
                        inf_uploaded_diff_model = gr.File(
                            label="Or Upload Diffusion Model (.pt)",
                            file_types=[".pt"]
                        )

                        inf_diffusion_config_path = gr.Dropdown(
                            label="Diffusion Config Path",
                            choices=get_available_diffusion_configs(),
                            value=get_available_diffusion_configs()[0] if get_available_diffusion_configs()[
                                                                              0] != "No diffusion config files found" else None,
                            info="Select diffusion config file"
                        )
                        inf_uploaded_diff_config = gr.File(
                            label="Or Upload Diffusion Config (.yaml/.yml)",
                            file_types=[".yaml", ".yml"]
                        )

                with gr.Column():
                    gr.Markdown("#### 🎛️ Diffusion Parameters")

                    inf_k_step = gr.Slider(
                        label="Diffusion Steps",
                        minimum=1,
                        maximum=1000,
                        value=100,
                        step=1,
                        info="Higher values = better quality but slower"
                    )

                    inf_only_diffusion = gr.Checkbox(
                        label="Only Diffusion Mode",
                        value=False,
                        info="Use only diffusion model without main model"
                    )

                    inf_second_encoding = gr.Checkbox(
                        label="Second Encoding",
                        value=False,
                        info="Re-encode audio before diffusion (experimental)"
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔊 Enhancement Settings")

                    inf_enhance = gr.Checkbox(
                        label="NSF-HiFiGAN Enhancement",
                        value=False,
                        info="May improve quality for small datasets"
                    )

                    inf_cluster_model_path = gr.Textbox(
                        label="Cluster Model Path",
                        placeholder="Path to cluster model or feature retrieval index",
                        info="Leave empty for auto-detection"
                    )

                    inf_cluster_infer_ratio = gr.Slider(
                        label="Cluster Inference Ratio",
                        minimum=0,
                        maximum=1,
                        value=0,
                        step=0.01,
                        info="Blend ratio for clustering/retrieval"
                    )

                    inf_feature_retrieval = gr.Checkbox(
                        label="Feature Retrieval",
                        value=False,
                        info="Use feature retrieval (disables clustering)"
                    )

                with gr.Column():
                    gr.Markdown("#### 🎚️ Mixing Settings")

                    inf_use_spk_mix = gr.Checkbox(
                        label="Speaker Mixing",
                        value=False,
                        info="Enable dynamic speaker mixing"
                    )

                    inf_loudness_envelope_adjustment = gr.Slider(
                        label="Loudness Envelope Adjustment",
                        minimum=0,
                        maximum=1,
                        value=1.0,
                        step=0.01,
                        info="1.0 = use output envelope, 0.0 = use input envelope"
                    )

            gr.Markdown("---")
            with gr.Row():
                inference_btn = gr.Button("🚀 Start Voice Conversion", variant="primary", scale=3)
                stop_inference_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            inference_status = gr.Textbox(label="Inference Status", interactive=False)

            with gr.Row():
                with gr.Column():
                    output_audio = gr.Audio(
                        label="Converted Audio - Preview",
                        interactive=False,
                    )
                with gr.Column():
                    output_download = gr.File(
                        label="Download Converted Audio",
                        interactive=False,
                        visible=False
                    )

            gr.Markdown("### 📊 Inference Logs")
            with gr.Row():
                inference_refresh_btn = gr.Button("🔄 Refresh Logs", scale=1)
                inference_clear_btn = gr.Button("🗑️ Clear Logs", scale=1, variant="secondary")

            inference_log_display = gr.Textbox(
                label="Voice Conversion Logs",
                lines=15,
                max_lines=20,
                interactive=False,
                elem_id="inference_log_display",
                info="Shows real-time logs for voice conversion process"
            )

    scroll_js = """
    setTimeout(() => {
        ['preprocess_log_display', 'train_log_display', 'diff_log_display', 'inference_log_display'].forEach(id => {
            const textarea = document.querySelector('#' + id + ' textarea');
            if (textarea) {
                textarea.scrollTop = textarea.scrollHeight;
                if (id === 'diff_log_display') {
                    console.log('Scrolling diff_log_display to bottom');
                    setTimeout(() => {
                        textarea.scrollTop = textarea.scrollHeight;
                    }, 100);
                    setTimeout(() => {
                        textarea.scrollTop = textarea.scrollHeight;
                    }, 300);
                }
            }
        });
    }, 200);
    """

    resample_btn.click(
        resample_audio,
        inputs=[dataset_raw_path, dataset_44k_path, skip_loudnorm],
        outputs=[resample_status, preprocess_log_display],
        js=scroll_js
    )

    config_btn.click(
        preprocess_config,
        inputs=[dataset_44k_path, speech_encoder, vol_aug, tiny_model],
        outputs=[config_status, preprocess_log_display],
        js=scroll_js
    )

    extract_btn.click(
        extract_features,
        inputs=[dataset_44k_path, f0_predictor, use_diff, num_processes, device_choice],
        outputs=[extract_status, preprocess_log_display],
        js=scroll_js
    )

    check_dataset_btn.click(
        lambda path: check_dataset_structure(path)[1],
        inputs=[dataset_raw_path],
        outputs=[dataset_info]
    )

    train_btn.click(
        start_training,
        inputs=[model_name, use_ascend],
        outputs=[train_status, train_log_display],
        js=scroll_js
    )

    stop_btn.click(
        stop_training_main,
        outputs=[train_status, train_log_display],
        js=scroll_js
    )

    diff_train_btn.click(
        start_diff_training,
        inputs=[diff_use_ascend],
        outputs=[diff_train_status, diff_log_display],
        js=scroll_js
    )

    diff_stop_btn.click(
        stop_training_diff,
        outputs=[diff_train_status, diff_log_display],
        js=scroll_js
    )

    test_monitor_btn.click(
        test_diffusion_monitoring,
        outputs=[diff_train_status, diff_log_display],
        js=scroll_js
    )


    def refresh_all_lists():
        return (
            gr.Dropdown(choices=get_available_models(),
                        value=get_available_models()[0] if get_available_models()[0] != "No models found" else None),
            gr.Dropdown(choices=get_available_configs(), value=get_available_configs()[0] if get_available_configs()[
                                                                                                 0] != "No config files found" else None),
            gr.Dropdown(choices=get_available_diffusion_models(),
                        value=get_available_diffusion_models()[0] if get_available_diffusion_models()[
                                                                         0] != "No diffusion models found" else None),
            gr.Dropdown(choices=get_available_diffusion_configs(),
                        value=get_available_diffusion_configs()[0] if get_available_diffusion_configs()[
                                                                          0] != "No diffusion config files found" else None)
        )


    refresh_models_btn.click(
        refresh_all_lists,
        outputs=[inf_model_path, inf_config_path, inf_diffusion_model_path, inf_diffusion_config_path]
    )


    def update_speakers(config_path):
        return gr.Dropdown(choices=get_speakers_from_config(config_path))


    inf_config_path.change(
        update_speakers,
        inputs=[inf_config_path],
        outputs=[inf_speaker]
    )

    inference_btn.click(
        run_inference,
        inputs=[
            inf_model_path, inf_uploaded_model, inf_config_path, inf_uploaded_config,
            inf_input_audio, inf_uploaded_audio,
            inf_trans, inf_speaker,
            inf_clip_duration, inf_linear_gradient, inf_f0_predictor, inf_auto_predict_f0,
            inf_slice_db, inf_device_selection, inf_noise_scale, inf_pad_seconds, inf_wav_format,
            inf_linear_gradient_retain, inf_enhancer_adaptive_key, inf_f0_filter_threshold,
            inf_use_diffusion, inf_diffusion_model_path, inf_uploaded_diff_model,
            inf_diffusion_config_path, inf_uploaded_diff_config, inf_k_step, inf_only_diffusion, inf_second_encoding,
            inf_enhance, inf_cluster_model_path, inf_cluster_infer_ratio, inf_feature_retrieval,
            inf_use_spk_mix, inf_loudness_envelope_adjustment
        ],
        outputs=[output_audio, output_download, inference_status],
        js=scroll_js
    )

    preprocess_refresh_btn.click(
        lambda: training_manager.get_all_logs(),
        outputs=[preprocess_log_display],
        js="setTimeout(() => { const textarea = document.querySelector('#preprocess_log_display textarea'); if (textarea) textarea.scrollTop = textarea.scrollHeight; }, 200)"
    )
    preprocess_clear_btn.click(
        lambda: (training_manager.clear_logs(), "")[1],
        outputs=[preprocess_log_display]
    )

    train_refresh_btn.click(
        lambda: training_manager.get_all_logs(),
        outputs=[train_log_display],
        js="setTimeout(() => { const textarea = document.querySelector('#train_log_display textarea'); if (textarea) textarea.scrollTop = textarea.scrollHeight; }, 200)"
    )
    train_clear_btn.click(
        lambda: (training_manager.clear_logs(), "")[1],
        outputs=[train_log_display]
    )

    diff_refresh_btn.click(
        lambda: training_manager.get_all_logs(),
        outputs=[diff_log_display],
        js=scroll_js
    )
    diff_clear_btn.click(
        lambda: (training_manager.clear_logs(), "")[1],
        outputs=[diff_log_display]
    )

    inference_refresh_btn.click(
        lambda: training_manager.get_all_logs(),
        outputs=[inference_log_display],
        js="setTimeout(() => { const textarea = document.querySelector('#inference_log_display textarea'); if (textarea) textarea.scrollTop = textarea.scrollHeight; }, 200)"
    )
    inference_clear_btn.click(
        lambda: (training_manager.clear_logs(), "")[1],
        outputs=[inference_log_display]
    )

    log_timer = gr.Timer(value=0.8, active=True)
    log_timer.tick(
        update_logs_with_scroll,
        outputs=[preprocess_log_display, train_log_display, diff_log_display, inference_log_display],
        js=scroll_js
    )

    diff_timer = gr.Timer(value=1.0, active=True)
    diff_timer.tick(
        force_update_diff_logs,
        outputs=[diff_log_display],
        js="""
        setTimeout(() => {
            const textarea = document.querySelector('#diff_log_display textarea');
            if (textarea) {
                console.log('[DIFF] Force updating diffusion model log area');
                textarea.scrollTop = textarea.scrollHeight;
                textarea.style.display = 'none';
                textarea.offsetHeight;
                textarea.style.display = '';
            }
        }, 100);
        """
    )

    app.css = """
    #preprocess_log_display textarea, 
    #train_log_display textarea, 
    #diff_log_display textarea,
    #inference_log_display textarea {
        font-family: monospace;
        font-size: 12px;
        background-color: #1e1e1e;
        color: #ffffff;
        scroll-behavior: smooth;
    }

    .log-container {
        position: relative;
    }
    """

if __name__ == "__main__":
    os.makedirs("dataset_raw", exist_ok=True)
    os.makedirs("dataset/44k", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("raw", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )