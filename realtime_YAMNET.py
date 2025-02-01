import pyaudio
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
import time
import os
from datetime import datetime
import psutil
import GPUtil
from typing import Tuple, Optional

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Real-time Sound Event Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create placeholder for title
st.title("Real-time Sound Event Detection")

# Initialize session state variables if they don't exist
if 'history_spectrograms' not in st.session_state:
    st.session_state.history_spectrograms = deque(maxlen=11)  # 10 seconds + current
if 'history_predictions' not in st.session_state:
    st.session_state.history_predictions = deque(maxlen=11)  # 10 seconds + current
if 'history_timestamps' not in st.session_state:
    st.session_state.history_timestamps = deque(maxlen=11)  # 10 seconds + current
if 'is_running' not in st.session_state:
    st.session_state.is_running = True

# Add system monitoring and audio controls in sidebar
with st.sidebar:
    st.title("Controls")
    
    # System Monitoring Section
    st.subheader("System Statistics")
    
    # Create containers for updating stats
    cpu_metric = st.empty()
    memory_metric = st.empty()
    gpu_metric = st.empty()
    
    # Audio Controls Section
    st.subheader("Audio Controls")
    
    # Initialize gain in session state if not exists
    if 'input_gain' not in st.session_state:
        st.session_state.input_gain = 0.0
    
    # Gain slider
    st.session_state.input_gain = st.slider(
        "Input Gain",
        min_value=0.0,
        max_value=32.0,
        value=st.session_state.input_gain,
        step=0.5,
        format="%f dB"
    )
    
    # Add container for dB meter only
    db_meter = st.empty()
    
    # Pause/Resume button
    if st.button("Pause/Resume"):
        st.session_state.is_running = not st.session_state.is_running
    
    status_text = "🟢 Running" if st.session_state.is_running else "🔴 Paused"
    st.write(f"Status: {status_text}")

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        st.sidebar.success("GPU memory growth enabled")
    except RuntimeError as e:
        st.sidebar.error(f"Error enabling memory growth: {e}")

# Import YAMNet modules and configure GPU
import yamnet.params as params
import yamnet.yamnet as yamnet_model

tf.config.set_visible_devices(physical_devices, 'GPU')

# Load model and weights
@st.cache_resource
def load_model():
    with tf.device('/GPU:0'):
        model = yamnet_model.yamnet_frames_model(params)
        model.load_weights('yamnet/yamnet.h5')
    return model

yamnet = load_model()
yamnet_classes = yamnet_model.class_names('yamnet/yamnet_class_map.csv')

# Audio setup
frame_len = int(params.SAMPLE_RATE * 1)  # 1sec window
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=params.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frame_len)

# Create placeholder for visualizations
spec_plot = st.empty()
pred_plot = st.empty()
current_preds = st.empty()

# Add session state for audio stream
if 'audio_stream' not in st.session_state:
    st.session_state.audio_stream = None
if 'audio_instance' not in st.session_state:
    st.session_state.audio_instance = None

# Function to safely manage audio stream
def get_audio_stream():
    if st.session_state.audio_stream is None:
        try:
            st.session_state.audio_instance = pyaudio.PyAudio()
            st.session_state.audio_stream = st.session_state.audio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=params.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frame_len
            )
        except Exception as e:
            st.error(f"Error initializing audio: {e}")
            return None
    return st.session_state.audio_stream

def cleanup_audio():
    if st.session_state.audio_stream is not None:
        try:
            st.session_state.audio_stream.stop_stream()
            st.session_state.audio_stream.close()
        except Exception:
            pass
    if st.session_state.audio_instance is not None:
        try:
            st.session_state.audio_instance.terminate()
        except Exception:
            pass
    st.session_state.audio_stream = None
    st.session_state.audio_instance = None

def update_visualizations():
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Mel Spectrogram History', 'Top Predictions Over Time'),
        vertical_spacing=0.3,
        row_heights=[0.6, 0.4]
    )

    # Plot spectrogram history
    if st.session_state.history_spectrograms:
        # Combine spectrograms horizontally
        combined_spec = np.hstack([spec.T for spec in st.session_state.history_spectrograms])
        
        fig.add_trace(
            go.Heatmap(
                z=combined_spec,
                colorscale='Jet',
                showscale=False
            ),
            row=1, col=1
        )

    # Plot prediction history
    if st.session_state.history_predictions and st.session_state.history_timestamps:
        # Get top 5 classes across all history
        all_preds = np.vstack(st.session_state.history_predictions)
        top_classes_indices = np.unique(np.argsort(-all_preds, axis=1)[:, :5].flatten())
        
        for idx in top_classes_indices:
            values = [pred[idx] for pred in st.session_state.history_predictions]
            fig.add_trace(
                go.Scatter(
                    x=list(st.session_state.history_timestamps),
                    y=values,
                    name=yamnet_classes[idx],
                    mode='lines+markers'
                ),
                row=2, col=1
            )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    fig.update_yaxes(title_text="Mel Bands", row=1, col=1)

    return fig

# Add new functions for system monitoring and audio processing
def update_system_stats():
    """Update system statistics in the sidebar."""
    # CPU Usage
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_metric.markdown(f"""
        <div style="padding: 10px; background: rgba(46,204,113,0.1); border-radius: 5px;">
            <h5 style="margin: 0;">CPU Usage</h5>
            <div style="font-size: 20px; font-weight: bold; color: #2ecc71;">
                {cpu_percent:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Memory Usage
    memory = psutil.virtual_memory()
    memory_metric.markdown(f"""
        <div style="padding: 10px; background: rgba(52,152,219,0.1); border-radius: 5px;">
            <h5 style="margin: 0;">Memory Usage</h5>
            <div style="font-size: 20px; font-weight: bold; color: #3498db;">
                {memory.percent:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # GPU Usage (if available)
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]  # Get first GPU
            gpu_metric.markdown(f"""
                <div style="padding: 10px; background: rgba(155,89,182,0.1); border-radius: 5px;">
                    <h5 style="margin: 0;">GPU Usage</h5>
                    <div style="font-size: 20px; font-weight: bold; color: #9b59b6;">
                        {gpu.load*100:.1f}%
                    </div>
                    <div style="font-size: 12px;">
                        Temp: {gpu.temperature}°C<br>
                        Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except Exception:
        gpu_metric.markdown("GPU stats unavailable")

def process_audio_frame(frame_data: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """Process audio frame with gain."""
    # Apply input gain
    frame_data = frame_data * (10 ** (st.session_state.input_gain / 20))
    
    # Calculate current dB level
    rms = np.sqrt(np.mean(frame_data**2))
    db_level = 20 * np.log10(rms) if rms > 0 else -120
    
    return frame_data, db_level

def update_predictions_display(predictions, yamnet_classes):
    latest_prediction = predictions[-1]
    top5_i = np.argsort(latest_prediction)[::-1][:5]
    
    # Create markdown text instead of HTML
    pred_text = "### Current Top Predictions:\n\n"
    
    for i in top5_i:
        confidence = latest_prediction[i]
        # Use markdown table-like formatting
        pred_text += f"{yamnet_classes[i]}: {confidence:.3f}\n\n"
    
    return pred_text

try:
    stream = get_audio_stream()
    if stream is None:
        st.error("Failed to initialize audio stream")
        st.stop()

    while True:
        # Update system stats
        update_system_stats()
        
        if st.session_state.is_running:
            try:
                # Read audio data
                data = stream.read(frame_len, exception_on_overflow=False)
                frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)
                
                # Process audio with gain
                frame_data, db_level = process_audio_frame(frame_data)
                
                # Update dB meter
                if 'db_level' in locals():
                    # Create a visual meter with color coding
                    db_color = "#2ecc71" if db_level > -60 else "#3498db"  # Green for higher levels, blue for lower
                    meter_html = f"""
                        <div style="padding: 10px; background: rgba(52,152,219,0.1); border-radius: 5px;">
                            <h5 style="margin: 0;">Input Level</h5>
                            <div style="font-size: 24px; font-weight: bold; color: {db_color};">
                                {db_level:.1f} dB
                            </div>
                            <div style="width: 100%; height: 10px; background: #eee; border-radius: 5px; margin-top: 5px;">
                                <div style="width: {min(100, max(0, (db_level + 60) * 1.67))}%; height: 100%; 
                                     background: {db_color}; border-radius: 5px; transition: width 0.1s;">
                                </div>
                            </div>
                        </div>
                    """
                    db_meter.markdown(meter_html, unsafe_allow_html=True)

                # Prepare input tensor and run prediction
                input_tensor = tf.convert_to_tensor(np.reshape(frame_data, [1, -1]), dtype=tf.float32)
                
                with tf.device('/GPU:0'):
                    scores, melspec = yamnet.predict(input_tensor, steps=1)
                    prediction = tf.reduce_mean(scores, axis=0).numpy()

                # Update history
                current_time = datetime.now().strftime('%H:%M:%S.%f')[:-4]
                st.session_state.history_spectrograms.append(melspec)
                st.session_state.history_predictions.append(prediction)
                st.session_state.history_timestamps.append(current_time)

            except Exception as e:
                st.error(f"Error processing audio: {e}")
                cleanup_audio()
                st.stop()

        # Always update visualizations, whether running or paused
        if st.session_state.history_spectrograms:
            fig = update_visualizations()
            
            # Key generation strategy
            if st.session_state.is_running:
                # Dynamic key for running state (changes every frame)
                plot_key = f"running_plot_{time.time()}"
                # Clear any existing pause key when running
                if 'pause_plot_key' in st.session_state:
                    del st.session_state.pause_plot_key
            else:
                # Static key for entire pause duration
                if 'pause_plot_key' not in st.session_state:
                    # Create unique key using first pause timestamp
                    st.session_state.pause_plot_key = f"paused_plot_{time.time()}"
                plot_key = st.session_state.pause_plot_key
            
            # Update plot with appropriate key
            spec_plot.plotly_chart(
                fig,
                use_container_width=False,
                width=1200,
                height=800,
                key=plot_key
            )

            # Handle predictions display with conditional updates
            if st.session_state.history_predictions:
                current_preds.markdown(
                    update_predictions_display(st.session_state.history_predictions, yamnet_classes)
                )

        # Small delay to prevent overwhelming the system
        time.sleep(0.1)

except Exception as e:
    st.error(f"Unexpected error: {e}")
finally:
    cleanup_audio()