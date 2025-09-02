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
    """Handler for monitoring diffusion model related log files"""

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
        """Read new content from log files"""
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
        # Add persistent log storage, limit max lines to prevent memory overflow
        self.log_history = deque(maxlen=2000)  # Keep maximum 2000 log entries
        self.log_lock = threading.Lock()  # For thread-safe log operations
        # Add training type identifier
        self.current_training_type = None  # 'main', 'diff', or None

        # Diffusion model log monitoring related
        self.diffusion_log_observer = None
        self.diffusion_log_handler = None
        self.diffusion_log_paths = [
            "logs/44k/diffusion/",
            "logs/diffusion/",
            "diffusion/logs/",
            "./logs/"
        ]

    def add_log(self, message):
        """Add log to history (thread-safe)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        with self.log_lock:
            self.log_history.append(formatted_message)
            self.log_queue.put(formatted_message)

    def add_diffusion_log(self, message):
        """Add diffusion model specific log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] 🌊 {message}"

        with self.log_lock:
            self.log_history.append(formatted_message)
            self.log_queue.put(formatted_message)

    def get_all_logs(self):
        """Get all historical logs"""
        with self.log_lock:
            return "\n".join(self.log_history)

    def get_new_logs(self):
        """Get new logs (without clearing history)"""
        new_logs = []
        while not self.log_queue.empty():
            try:
                new_logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return new_logs

    def clear_logs(self):
        """Clear log history"""
        with self.log_lock:
            self.log_history.clear()
            # Clear remaining logs in queue
            while not self.log_queue.empty():
                try:
                    self.log_queue.get_nowait()
                except queue.Empty:
                    break

    def setup_diffusion_monitoring(self):
        """Setup diffusion model log file monitoring"""
        try:
            if self.diffusion_log_observer is None:
                self.diffusion_log_handler = DiffusionLogHandler(self)
                self.diffusion_log_observer = Observer()

                # Monitor possible log directories
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

                # Also setup Python log interception
                self.setup_python_log_capture()

        except Exception as e:
            self.add_log(f"Error setting up diffusion monitoring: {str(e)}")

    def setup_python_log_capture(self):
        """Setup Python log capture to monitor specific modules"""
        try:
            # Create custom log handler
            class DiffusionLogCaptureHandler(logging.Handler):
                def __init__(self, training_manager):
                    super().__init__()
                    self.training_manager = training_manager

                def emit(self, record):
                    try:
                        # Check if log record is from target modules
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
                        pass  # Avoid exceptions in log handling affecting training

            # Add to root logger
            diffusion_handler = DiffusionLogCaptureHandler(self)
            diffusion_handler.setLevel(logging.DEBUG)

            # Get root logger and add handler
            root_logger = logging.getLogger()
            root_logger.addHandler(diffusion_handler)
            root_logger.setLevel(logging.DEBUG)

            self.add_log("🎯 Python log interceptor setup, monitoring saver.py and solver.py")

        except Exception as e:
            self.add_log(f"Error setting up Python log capture: {str(e)}")

    def stop_diffusion_monitoring(self):
        """Stop diffusion model log monitoring"""
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
        """Read process output and put into queue"""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_message = output.strip()

                # Process logs based on training type
                if self.current_training_type == 'diff':
                    log_message = f"[TRAIN_DIFF] {log_message}"
                elif self.current_training_type == 'main':
                    log_message = f"[MAIN] {log_message}"

                self.add_log(log_message)

    def run_command(self, cmd, cwd=None):
        """Run command and output logs in real time"""
        try:
            self.add_log(f"Executing command: {cmd}")

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True,
                cwd=cwd,
                env=dict(os.environ, PYTHONUNBUFFERED='1')  # Ensure Python output is not buffered
            )

            # Start thread to read output
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
        """Stop training process"""
        if self.current_process:
            try:
                self.add_log("Stopping training process...")

                # Get process and all its child processes
                parent = psutil.Process(self.current_process.pid)
                children = parent.children(recursive=True)

                # Terminate all child processes
                for child in children:
                    child.terminate()

                # Terminate parent process
                parent.terminate()

                # Wait for processes to end
                gone, still_alive = psutil.wait_procs(children + [parent], timeout=5)

                # Force kill if processes are still alive
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
                # If it's diffusion model training, stop monitoring
                if self.current_training_type == 'diff':
                    self.stop_diffusion_monitoring()


training_manager = TrainingManager()


def check_dataset_structure(dataset_path):
    """Check dataset structure"""
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
    """Resample audio"""
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
    """Generate configuration file"""
    training_manager.stop_flag = False
    progress(0, desc="Generating config file...")

    cmd = f"python preprocess_flist_config.py --source_dir {dataset_44k_path} --speech_encoder {speech_encoder}"
    if vol_aug:
        cmd += " --vol_aug"
    if tiny_model:
        cmd += " --tiny"

    success = training_manager.run_command(cmd)

    if success:
        # Read generated config file information
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
    """Extract features"""
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
    """Start training"""
    if training_manager.is_training:
        return "⚠️ Training already in progress", training_manager.get_all_logs()

    training_manager.is_training = True
    training_manager.stop_flag = False
    training_manager.current_training_type = 'main'

    # Ensure model directory exists
    model_dir = f"logs/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # Build training command according to README
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
    """Start diffusion model training"""
    if training_manager.is_training:
        return "⚠️ Training already in progress", training_manager.get_all_logs()

    training_manager.is_training = True
    training_manager.stop_flag = False
    training_manager.current_training_type = 'diff'

    # Check if config file exists
    config_path = "configs/diffusion.yaml"
    if not os.path.exists(config_path):
        training_manager.add_log(f"❌ Config file does not exist: {config_path}")
        training_manager.add_log("Please complete preprocessing steps to generate config file")
        training_manager.is_training = False
        training_manager.current_training_type = None
        return "❌ Config file does not exist", training_manager.get_all_logs()

    # Build training command according to README
    if use_ascend:
        cmd = f"python train_diff_ascend.py -c configs/diffusion.yaml"
    else:
        cmd = f"python train_diff.py -c configs/diffusion.yaml"

    def train_thread():
        training_manager.add_log(f"🌊 Starting diffusion model training")
        training_manager.add_log(f"Using device: {'Ascend NPU' if use_ascend else 'NVIDIA GPU'}")
        training_manager.add_log(f"Diffusion model will be saved to: logs/44k/diffusion")

        # Setup diffusion model specific log monitoring
        training_manager.setup_diffusion_monitoring()
        training_manager.add_log(f"🔍 Started capturing output from diffusion/logger/saver.py and diffusion/solver.py")
        training_manager.add_log(f"Starting diffusion model training process...")

        success = training_manager.run_command(cmd)
        training_manager.is_training = False
        training_manager.current_training_type = None

        # Stop log monitoring
        training_manager.stop_diffusion_monitoring()

        if success:
            training_manager.add_log("✅ Diffusion model training completed!")
        else:
            training_manager.add_log("❌ Diffusion model training failed or interrupted")

    thread = threading.Thread(target=train_thread)
    thread.daemon = True
    thread.start()

    return "🚀 Diffusion model training started", training_manager.get_all_logs()


def stop_training():
    """Stop training"""
    if not training_manager.is_training:
        training_manager.add_log("⚠️ No training currently in progress")
        return "⚠️ No training currently in progress", training_manager.get_all_logs()

    training_manager.stop_training()
    return "⏹️ Stopping training...", training_manager.get_all_logs()


def stop_training_main():
    """Stop main model training"""
    return stop_training()


def stop_training_diff():
    """Stop diffusion model training"""
    return stop_training()


def update_logs():
    """Update log display - return complete historical logs"""
    return training_manager.get_all_logs()


def update_logs_with_scroll():
    """Update log display and trigger scroll"""
    logs = training_manager.get_all_logs()
    # Add some statistics for debugging
    log_count = len(training_manager.log_history)
    if log_count > 0:
        # Output debug info to browser console
        print(f"[DEBUG] Updating logs: {log_count} total logs")
    return logs, logs, logs  # Return to three different log display areas


def force_update_diff_logs():
    """Force update diffusion model logs"""
    logs = training_manager.get_all_logs()
    print(f"[DEBUG] Force updating diffusion model logs: {len(logs)} characters")
    return logs


def clear_logs():
    """Clear logs"""
    training_manager.clear_logs()
    return ""


def test_diffusion_monitoring():
    """Test diffusion model monitoring functionality"""
    training_manager.setup_diffusion_monitoring()
    training_manager.add_log("🧪 Diffusion model monitoring test started")
    training_manager.add_log("📝 Now will capture output from saver.py and solver.py")
    return "🧪 Test monitoring started", training_manager.get_all_logs()


# Create Gradio interface
with gr.Blocks(title="So-VITs-SVC-Fix WebUI") as app:
    gr.Markdown("""
    # So-VITs-SVC-Fix WebUI

    ### Usage Flow:
    1. **Data Preparation**: Put audio files into `dataset_raw/speaker_name/` folders
    2. **Preprocessing**: Execute in order: Resample → Generate Config → Extract Features
    3. **Training**: Configure parameters and start training
    4. **Monitoring**: View real-time logs to understand training progress
    """)

    with gr.Tabs():
        # Data preprocessing tab
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

            # Preprocessing log area
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

        # Training configuration tab
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

            # Main model training log area
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

        # Diffusion model training tab
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

                    # Add test button
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

            # Diffusion model training log area
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

    # Bind event handlers - add scroll JS
    scroll_js = """
    setTimeout(() => {
        ['preprocess_log_display', 'train_log_display', 'diff_log_display'].forEach(id => {
            const textarea = document.querySelector('#' + id + ' textarea');
            if (textarea) {
                textarea.scrollTop = textarea.scrollHeight;
                // Extra handling for diffusion model log area
                if (id === 'diff_log_display') {
                    console.log('Scrolling diff_log_display to bottom');
                    // Multiple attempts to ensure scrolling succeeds
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

    # Preprocessing related events
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

    # Main model training related events
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

    # Diffusion model training related events
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

    # Test monitoring functionality
    test_monitor_btn.click(
        test_diffusion_monitoring,
        outputs=[diff_train_status, diff_log_display],
        js=scroll_js
    )

    # Log refresh and clear events - handle different areas separately
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

    # Timer to update all log display areas
    log_timer = gr.Timer(value=0.8, active=True)
    log_timer.tick(
        update_logs_with_scroll,
        outputs=[preprocess_log_display, train_log_display, diff_log_display],
        js=scroll_js
    )

    # Add dedicated timer for diffusion model
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
                // Trigger redraw
                textarea.style.display = 'none';
                textarea.offsetHeight; // Trigger reflow
                textarea.style.display = '';
            }
        }, 100);
        """
    )

    # Add custom CSS to ensure auto-scroll
    app.css = """
    #preprocess_log_display textarea, 
    #train_log_display textarea, 
    #diff_log_display textarea {
        font-family: monospace;
        font-size: 12px;
        background-color: #1e1e1e;
        color: #ffffff;
        scroll-behavior: smooth;
    }

    /* Ensure log areas auto-scroll to bottom */
    .log-container {
        position: relative;
    }
    """

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs("dataset_raw", exist_ok=True)
    os.makedirs("dataset/44k", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Launch WebUI
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )