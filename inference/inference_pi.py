#!/usr/bin/env python3
"""
Real-Time Parkinson's Detection System for Raspberry Pi
WITH POWER CONSUMPTION MONITORING AND COMPREHENSIVE TESTING
Processes streaming sensor data with sliding window inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
import subprocess
from collections import deque
from datetime import datetime
from scipy.signal import butter, sosfilt, sosfilt_zi
import math
import json
import glob

# ============================================================================
# Model Architecture
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        timestep = x.size(1)
        return x + self.pe[:timestep, :].unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.linear2 = nn.Linear(d_ff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x + residual)
        return x


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention_1to2 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_2to1 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_1 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention_2 = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm_cross_1 = nn.LayerNorm(model_dim)
        self.norm_cross_2 = nn.LayerNorm(model_dim)
        self.norm_self_1 = nn.LayerNorm(model_dim)
        self.norm_self_2 = nn.LayerNorm(model_dim)
        
        self.feed_forward_1 = FeedForward(model_dim, d_ff, dropout)
        self.feed_forward_2 = FeedForward(model_dim, d_ff, dropout)
        
    def forward(self, channel_1, channel_2):
        channel_1_cross_attn, _ = self.cross_attention_1to2(
            query=channel_1, key=channel_2, value=channel_2
        )
        channel_1_cross = self.norm_cross_1(channel_1 + channel_1_cross_attn)
        
        channel_2_cross_attn, _ = self.cross_attention_2to1(
            query=channel_2, key=channel_1, value=channel_1
        )
        channel_2_cross = self.norm_cross_2(channel_2 + channel_2_cross_attn)
        
        channel_1_self_attn, _ = self.self_attention_1(
            query=channel_1_cross, key=channel_1_cross, value=channel_1_cross
        )
        channel_1_self = self.norm_self_1(channel_1_cross + channel_1_self_attn)
        
        channel_2_self_attn, _ = self.self_attention_2(
            query=channel_2_cross, key=channel_2_cross, value=channel_2_cross
        )
        channel_2_self = self.norm_self_2(channel_2_cross + channel_2_self_attn)
        
        channel_1_out = self.feed_forward_1(channel_1_self)
        channel_2_out = self.feed_forward_2(channel_2_self)

        return channel_1_out, channel_2_out


class MainModel(nn.Module):
    def __init__(self, input_dim=6, model_dim=128, num_heads=8, num_layers=4, 
                 d_ff=512, dropout=0.1, timestep=256, num_classes=2, fusion_method='concat'):
        super().__init__()
        
        self.model_dim = model_dim
        self.timestep = timestep
        self.fusion_method = fusion_method
        
        self.left_projection = nn.Linear(input_dim, model_dim)
        self.right_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len=timestep)
        
        self.layers = nn.ModuleList([
            CrossAttention(model_dim, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        fusion_dim = model_dim * 2

        self.head_hc_vs_pd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)
        )
        
        self.head_pd_vs_dd = nn.Sequential(
            nn.Linear(fusion_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 2)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, left_wrist, right_wrist, device=None):
        left_encoded = self.left_projection(left_wrist)
        right_encoded = self.right_projection(right_wrist)

        left_encoded = self.positional_encoding(left_encoded)
        right_encoded = self.positional_encoding(right_encoded)
        
        left_encoded = self.dropout(left_encoded)
        right_encoded = self.dropout(right_encoded)
        
        for layer in self.layers:
            left_encoded, right_encoded = layer(left_encoded, right_encoded)

        left_pool = self.global_pool(left_encoded.transpose(1, 2)).squeeze(-1)
        right_pool = self.global_pool(right_encoded.transpose(1, 2)).squeeze(-1)

        fused_features = torch.cat([left_pool, right_pool], dim=1)

        logits_hc_vs_pd = self.head_hc_vs_pd(fused_features)
        logits_pd_vs_dd = self.head_pd_vs_dd(fused_features)

        return logits_hc_vs_pd, logits_pd_vs_dd


# ============================================================================
# Power Monitoring
# ============================================================================

class PowerMonitor:
    """Monitor power consumption metrics on Raspberry Pi"""
    
    def __init__(self):
        self.is_raspberry_pi = self._check_raspberry_pi()
        self.monitoring = False
        self.measurements = []
        
    def _check_raspberry_pi(self):
        """Check if running on Raspberry Pi"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'Raspberry Pi' in model
        except:
            return False
    
    def _get_cpu_temp(self):
        """Get CPU temperature in Celsius"""
        try:
            if self.is_raspberry_pi:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read().strip()) / 1000.0
                    return temp
            else:
                # Fallback for non-Pi systems
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            return entries[0].current
        except:
            pass
        return None
    
    def _get_cpu_voltage(self):
        """Get CPU voltage using vcgencmd"""
        if not self.is_raspberry_pi:
            return None
        try:
            result = subprocess.run(['vcgencmd', 'measure_volts', 'core'], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                voltage_str = result.stdout.strip()
                voltage = float(voltage_str.split('=')[1].replace('V', ''))
                return voltage
        except:
            pass
        return None
    
    def _get_cpu_freq(self):
        """Get CPU frequency in MHz"""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return freq.current
        except:
            pass
        return None
    
    def get_metrics(self):
        """Get current power/performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_temp': self._get_cpu_temp(),
            'cpu_voltage': self._get_cpu_voltage(),
            'cpu_freq_mhz': self._get_cpu_freq(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / (1024 * 1024)
        }
        return metrics
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring = True
        self.measurements = []
    
    def record_measurement(self):
        """Record a single measurement"""
        if self.monitoring:
            metrics = self.get_metrics()
            self.measurements.append(metrics)
            return metrics
        return None
    
    def stop_monitoring(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        
        if not self.measurements:
            return None
        
        stats = {
            'num_measurements': len(self.measurements),
            'duration_seconds': self.measurements[-1]['timestamp'] - self.measurements[0]['timestamp']
        }
        
        # Calculate averages and peaks
        for key in ['cpu_percent', 'cpu_temp', 'cpu_voltage', 'cpu_freq_mhz', 'memory_percent', 'memory_mb']:
            values = [m[key] for m in self.measurements if m[key] is not None]
            if values:
                stats[f'{key}_avg'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
        
        # Estimate power consumption (rough approximation for Pi)
        if stats.get('cpu_voltage_avg') and stats.get('cpu_percent_avg'):
            # Very rough estimate: assume base current of 500mA + variable load
            base_current = 0.5  # Amperes
            load_current = (stats['cpu_percent_avg'] / 100.0) * 1.5  # Up to 1.5A under load
            total_current = base_current + load_current
            estimated_power_watts = stats['cpu_voltage_avg'] * total_current
            stats['estimated_power_watts'] = estimated_power_watts
            stats['estimated_energy_joules'] = estimated_power_watts * stats['duration_seconds']
        
        return stats


# ============================================================================
# Real-Time Filter
# ============================================================================

class RealtimeBandpassFilter:
    """Causal bandpass filter for real-time processing"""
    
    def __init__(self, fs=64, lowcut=0.1, highcut=20, order=5, n_channels=6):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.n_channels = n_channels
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.sos = butter(order, [low, high], btype='band', output='sos')
        
        self.zi = [sosfilt_zi(self.sos) for _ in range(n_channels)]
    
    def filter(self, sample):
        """Filter a single sample across all channels"""
        filtered_sample = np.zeros_like(sample)
        
        for ch in range(self.n_channels):
            filtered_sample[ch], self.zi[ch] = sosfilt(
                self.sos, [sample[ch]], zi=self.zi[ch]
            )
        
        return filtered_sample
    
    def reset(self):
        """Reset filter states"""
        self.zi = [sosfilt_zi(self.sos) for _ in range(self.n_channels)]


# ============================================================================
# Sliding Window Buffer
# ============================================================================

class SlidingWindowBuffer:
    """Circular buffer for maintaining sliding window of sensor data"""
    
    def __init__(self, window_size=256, n_channels=6, overlap=0.5):
        self.window_size = window_size
        self.n_channels = n_channels
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
        self.buffer = deque(maxlen=window_size)
        self.sample_count = 0
        
    def add_sample(self, sample):
        """Add a new sample to the buffer"""
        self.buffer.append(sample)
        self.sample_count += 1
        
    def is_ready(self):
        """Check if buffer has enough samples for inference"""
        return len(self.buffer) >= self.window_size
    
    def should_infer(self):
        """Check if we should run inference"""
        if not self.is_ready():
            return False
        
        return (self.sample_count - self.window_size) % self.step_size == 0
    
    def get_window(self):
        """Get current window as numpy array"""
        if not self.is_ready():
            return None
        return np.array(list(self.buffer))
    
    def reset(self):
        """Reset buffer"""
        self.buffer.clear()
        self.sample_count = 0


# ============================================================================
# Real-Time Inference Engine
# ============================================================================

class RealtimeInferenceEngine:
    """Engine for real-time inference on streaming sensor data"""
    
    def __init__(self, model, device='cpu', window_size=256, overlap=0.5,
                 apply_filter=True, fs=64, downsample_rate=None, power_monitor=None):
        self.model = model
        self.device = device
        self.window_size = window_size
        self.overlap = overlap
        self.apply_filter = apply_filter
        self.fs = fs
        self.downsample_rate = downsample_rate
        self.power_monitor = power_monitor
        
        self.left_buffer = SlidingWindowBuffer(window_size, n_channels=6, overlap=overlap)
        self.right_buffer = SlidingWindowBuffer(window_size, n_channels=6, overlap=overlap)
        
        if apply_filter:
            self.left_filter = RealtimeBandpassFilter(fs=fs, n_channels=6)
            self.right_filter = RealtimeBandpassFilter(fs=fs, n_channels=6)
        
        self.downsample_counter = 0
        
        self.inference_count = 0
        self.total_inference_time = 0
        self.last_prediction = None
        self.prediction_history = deque(maxlen=10)
        
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with dummy data"""
        print("Warming up model...")
        dummy_left = torch.randn(1, self.window_size, 6).to(self.device)
        dummy_right = torch.randn(1, self.window_size, 6).to(self.device)
        
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy_left, dummy_right, self.device)
        
        print("Model ready!")
    
    def process_sample(self, left_sample, right_sample):
        """Process a single sample from both wrists"""
        if self.downsample_rate is not None:
            self.downsample_counter += 1
            if self.downsample_counter % self.downsample_rate != 0:
                return None
        
        if self.apply_filter:
            left_sample = self.left_filter.filter(left_sample)
            right_sample = self.right_filter.filter(right_sample)
        
        self.left_buffer.add_sample(left_sample)
        self.right_buffer.add_sample(right_sample)
        
        if self.left_buffer.should_infer() and self.right_buffer.should_infer():
            return self._run_inference()
        
        return None
    
    def _run_inference(self):
        """Run inference on current window"""
        left_window = self.left_buffer.get_window()
        right_window = self.right_buffer.get_window()
        
        if left_window is None or right_window is None:
            return None
        
        left_tensor = torch.FloatTensor(left_window).unsqueeze(0).to(self.device)
        right_tensor = torch.FloatTensor(right_window).unsqueeze(0).to(self.device)
        
        # Record power before inference
        if self.power_monitor:
            pre_power = self.power_monitor.record_measurement()
        
        start_time = time.time()
        
        with torch.no_grad():
            hc_pd_logits, pd_dd_logits = self.model(left_tensor, right_tensor, self.device)
            
            hc_pd_probs = F.softmax(hc_pd_logits, dim=1)
            pd_dd_probs = F.softmax(pd_dd_logits, dim=1)
            
            hc_pd_pred = torch.argmax(hc_pd_probs, dim=1).item()
            pd_dd_pred = torch.argmax(pd_dd_probs, dim=1).item()
            
            hc_pd_conf = hc_pd_probs.max(dim=1)[0].item()
            pd_dd_conf = pd_dd_probs.max(dim=1)[0].item()
        
        inference_time = time.time() - start_time
        
        # Record power after inference
        if self.power_monitor:
            post_power = self.power_monitor.record_measurement()
        
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        prediction = {
            'timestamp': datetime.now(),
            'inference_num': self.inference_count,
            'hc_vs_pd': {
                'prediction': 'Healthy' if hc_pd_pred == 0 else 'Parkinsons',
                'confidence': hc_pd_conf * 100,
                'probs': hc_pd_probs.cpu().numpy()[0]
            },
            'pd_vs_dd': {
                'prediction': 'Parkinsons' if pd_dd_pred == 0 else 'Other Disorder',
                'confidence': pd_dd_conf * 100,
                'probs': pd_dd_probs.cpu().numpy()[0]
            },
            'performance': {
                'inference_time_ms': inference_time * 1000,
                'avg_inference_time_ms': (self.total_inference_time / self.inference_count) * 1000,
                'buffer_size': len(self.left_buffer.buffer)
            }
        }
        
        self.last_prediction = prediction
        self.prediction_history.append(prediction)
        
        return prediction
    
    def get_stats(self):
        """Get engine statistics"""
        avg_time = 0
        if self.inference_count > 0:
            avg_time = (self.total_inference_time / self.inference_count * 1000)
        
        return {
            'total_inferences': self.inference_count,
            'avg_inference_time_ms': avg_time,
            'buffer_fill_left': len(self.left_buffer.buffer),
            'buffer_fill_right': len(self.right_buffer.buffer),
            'last_prediction': self.last_prediction
        }
    
    def reset(self):
        """Reset engine state"""
        self.left_buffer.reset()
        self.right_buffer.reset()
        if self.apply_filter:
            self.left_filter.reset()
            self.right_filter.reset()
        self.downsample_counter = 0


# ============================================================================
# Sensor Data Simulator
# ============================================================================

class SensorDataSimulator:
    """Simulates streaming sensor data from pre-recorded files"""
    
    def __init__(self, left_data_path, right_data_path, sample_rate=100, 
                 real_time=True, skip_samples=50):
        self.left_data = np.loadtxt(left_data_path, delimiter=",")
        self.right_data = np.loadtxt(right_data_path, delimiter=",")
        
        self.left_data = self.left_data[:, :6]
        self.right_data = self.right_data[:, :6]
        
        if skip_samples > 0:
            self.left_data = self.left_data[skip_samples:, :]
            self.right_data = self.right_data[skip_samples:, :]
        
        self.sample_rate = sample_rate
        self.real_time = real_time
        self.sample_interval = 1.0 / sample_rate
        
        self.current_index = 0
        self.total_samples = min(len(self.left_data), len(self.right_data))
        
        print(f"  Loaded {self.total_samples} samples at {sample_rate}Hz")
    
    def get_next_sample(self):
        """Get next sample from both sensors"""
        if self.current_index >= self.total_samples:
            return None, None
        
        left_sample = self.left_data[self.current_index]
        right_sample = self.right_data[self.current_index]
        
        self.current_index += 1
        
        if self.real_time:
            time.sleep(self.sample_interval)
        
        return left_sample, right_sample
    
    def has_data(self):
        """Check if more data is available"""
        return self.current_index < self.total_samples
    
    def reset(self):
        """Reset simulator"""
        self.current_index = 0


# ============================================================================
# Main Detection System
# ============================================================================

class RealtimeDetectionSystem:
    """Complete real-time detection system"""
    
    def __init__(self, model_path, device='cpu', window_size=256, overlap=0.5,
                 apply_filter=True, target_fs=64, verbose=True):
        self.device = device
        self.verbose = verbose
        self.target_fs = target_fs
        
        self.model = self._load_model(model_path)
        self.power_monitor = PowerMonitor()
        
        self.downsample_rate = int(100 / target_fs) if target_fs < 100 else None
        
        self.engine = RealtimeInferenceEngine(
            model=self.model,
            device=device,
            window_size=window_size,
            overlap=overlap,
            apply_filter=apply_filter,
            fs=target_fs,
            downsample_rate=self.downsample_rate,
            power_monitor=self.power_monitor
        )
        
        self.all_predictions = []
        self.start_time = None
        
    def _load_model(self, model_path):
        """Load model from checkpoint"""
        if self.verbose:
            print(f"Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'hyperparameters' in checkpoint:
            params = checkpoint['hyperparameters']
            model = MainModel(
                input_dim=6,
                model_dim=params['model_dim'],
                num_heads=params['num_heads'],
                num_layers=params['num_layers'],
                d_ff=params['d_ff'],
                dropout=params['dropout'],
                timestep=256,
                num_classes=2
            )
        elif 'config' in checkpoint:
            config = checkpoint['config']
            model = MainModel(
                input_dim=config.get('input_dim', 6),
                model_dim=config.get('model_dim', 128),
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 4),
                d_ff=config.get('d_ff', 512),
                dropout=config.get('dropout', 0.1),
                timestep=config.get('timestep', 256),
                num_classes=config.get('num_classes', 2)
            )
        else:
            model = MainModel(
                input_dim=6,
                model_dim=32,
                num_heads=8,
                num_layers=3,
                d_ff=256,
                dropout=0.12281570220908891,
                timestep=256,
                num_classes=2
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        if self.verbose:
            print("Model loaded successfully!")
        
        return model
    
    def process_stream(self, data_source, task_name="Unknown"):
        """Process streaming data from a source"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TASK: {task_name}")
            print(f"{'='*80}")
        
        self.start_time = time.time()
        self.power_monitor.start_monitoring()
        
        sample_count = 0
        
        try:
            while data_source.has_data():
                left_sample, right_sample = data_source.get_next_sample()
                
                if left_sample is None or right_sample is None:
                    break
                
                sample_count += 1
                
                # Record power periodically
                if sample_count % 50 == 0:
                    self.power_monitor.record_measurement()
                
                prediction = self.engine.process_sample(left_sample, right_sample)
                
                if prediction is not None:
                    self.all_predictions.append(prediction)
                    if self.verbose:
                        self._display_prediction(prediction)
        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        power_stats = self.power_monitor.stop_monitoring()
        
        task_results = self._generate_task_results(sample_count, power_stats, task_name)
        system.save_task_results(task_results)
        
        if self.verbose:
            self._display_task_summary(task_results)
        
        # Reset for next task
        self.engine.reset()
        self.all_predictions = []
        
        return task_results
    
    def _display_prediction(self, prediction):
        """Display prediction results"""
        ts = prediction['timestamp'].strftime('%H:%M:%S.%f')[:-3]
        inf_num = prediction['inference_num']
        
        hc_pd = prediction['hc_vs_pd']
        perf = prediction['performance']
        
        print(f"[{ts}] Inference #{inf_num}: {hc_pd['prediction']:12s} "
              f"({hc_pd['confidence']:5.1f}%) | {perf['inference_time_ms']:.1f}ms")
    
    def _generate_task_results(self, sample_count, power_stats, task_name):
        """Generate comprehensive results for a task"""
        stats = self.engine.get_stats()
        elapsed = time.time() - self.start_time
        
        results = {
            'task_name': task_name,
            'total_samples': sample_count,
            'total_inferences': stats['total_inferences'],
            'duration_seconds': elapsed,
            'avg_sample_rate_hz': sample_count / elapsed if elapsed > 0 else 0,
            'timing': {
                'avg_inference_time_ms': stats['avg_inference_time_ms'],
                'total_inference_time_ms': self.engine.total_inference_time * 1000
            },
            'power': power_stats if power_stats else {},
            'predictions': {},
            
            # ADD THESE NEW FIELDS FOR DETAILED DATA:
            'detailed_predictions': [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'inference_num': p['inference_num'],
                    'hc_vs_pd_prediction': p['hc_vs_pd']['prediction'],
                    'hc_vs_pd_confidence': p['hc_vs_pd']['confidence'],
                    'hc_vs_pd_probs': p['hc_vs_pd']['probs'].tolist(),
                    'pd_vs_dd_prediction': p['pd_vs_dd']['prediction'],
                    'pd_vs_dd_confidence': p['pd_vs_dd']['confidence'],
                    'pd_vs_dd_probs': p['pd_vs_dd']['probs'].tolist(),
                    'inference_time_ms': p['performance']['inference_time_ms']
                }
                for p in self.all_predictions
            ],
            
            'detailed_power_measurements': [
                {
                    'timestamp': m['timestamp'],
                    'cpu_percent': m['cpu_percent'],
                    'cpu_temp': m['cpu_temp'],
                    'cpu_voltage': m['cpu_voltage'],
                    'cpu_freq_mhz': m['cpu_freq_mhz'],
                    'memory_percent': m['memory_percent'],
                    'memory_mb': m['memory_mb']
                }
                for m in self.power_monitor.measurements
            ]
        }
    
        return results
    def save_task_results(self, results, output_dir="results"):
        """Save detailed task results to JSON file"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{results['task_name']}_results.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to: {filepath}")
    
    def _display_task_summary(self, results):
        """Display task summary"""
        print(f"\n{'─'*80}")
        print(f"SUMMARY - {results['task_name']}")
        print(f"{'─'*80}")
        print(f"Samples: {results['total_samples']} | Inferences: {results['total_inferences']} | Time: {results['duration_seconds']:.2f}s")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import glob
    
    print("="*80)
    print("Real-Time Parkinson's Detection System - ALL TASKS")
    print("="*80)
    
    # Find all task pairs
    left_files = sorted(glob.glob("*_LeftWrist.txt"))
    
    print(f"\nFound {len(left_files)} tasks to process\n")
    
    # Create detection system once
    system = RealtimeDetectionSystem(
        model_path="best_model.pth",
        device='cpu',
        window_size=256,
        overlap=0.5,
        apply_filter=True,
        target_fs=64,
        verbose=True
    )
    
    # Process each task
    for left_file in left_files:
        right_file = left_file.replace("_LeftWrist.txt", "_RightWrist.txt")
        task_name = left_file.replace("001_", "").replace("_LeftWrist.txt", "")
        
        # Create simulator
        simulator = SensorDataSimulator(
            left_data_path=left_file,
            right_data_path=right_file,
            sample_rate=100,
            real_time=False,  # Fast processing
            skip_samples=50
        )
        
        # Run detection
        system.process_stream(simulator, task_name=task_name)
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETE!")
    print("="*80)