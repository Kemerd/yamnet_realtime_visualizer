"""
Real-time noise reduction using Dual-task Learning Network (DTLN)
Implements a two-stage enhancement:
1. Frequency domain processing (magnitude mask estimation)
2. Time domain processing (waveform enhancement)
"""

import numpy as np
import sounddevice as sd
import tensorflow as tf
import queue
import time

class DTLNNoiseReducer:
    """
    Real-time noise reduction using DTLN TensorFlow Lite models
    Uses a two-stage approach:
    1. Model 1: Frequency domain processing for mask estimation
    2. Model 2: Time domain processing for waveform enhancement
    """
    def __init__(self, sample_rate=16000):
        # Audio parameters
        self.sample_rate = sample_rate
        self.block_len_ms = 32  # Block length in milliseconds
        self.block_shift_ms = 8  # Block shift in milliseconds
        
        # Calculate block sizes
        self.block_len = int(np.round(self.sample_rate * (self.block_len_ms / 1000)))
        self.block_shift = int(np.round(self.sample_rate * (self.block_shift_ms / 1000)))
        
        # Initialize TFLite models
        self.interpreter_1 = tf.lite.Interpreter(model_path='models/model_quant_1.tflite')
        self.interpreter_1.allocate_tensors()
        self.interpreter_2 = tf.lite.Interpreter(model_path='models/model_quant_2.tflite')
        self.interpreter_2.allocate_tensors()
        
        # Get model details
        self.input_details_1 = self.interpreter_1.get_input_details()
        self.output_details_1 = self.interpreter_1.get_output_details()
        self.input_details_2 = self.interpreter_2.get_input_details()
        self.output_details_2 = self.interpreter_2.get_output_details()
        
        # Initialize states for LSTM layers
        self.states_1 = np.zeros(self.input_details_1[1]['shape']).astype('float32')
        self.states_2 = np.zeros(self.input_details_2[1]['shape']).astype('float32')
        
        # Initialize buffers
        self.in_buffer = np.zeros((self.block_len)).astype('float32')
        self.out_buffer = np.zeros((self.block_len)).astype('float32')
        
        # Processing queue
        self.input_queue = queue.Queue()
        
    def process_audio(self, indata, frames, time_info, status):
        """Callback for audio stream - queues audio for processing"""
        if status:
            print(status)
        self.input_queue.put(indata.copy())
        
    def _enhance_block(self, audio_block):
        """
        Enhance a single block of audio using both models
        Args:
            audio_block: numpy array of audio samples
        Returns:
            Enhanced audio block
        """
        # Stage 1: Frequency domain processing
        in_block_fft = np.fft.rfft(audio_block)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        
        # Reshape magnitude for model input
        in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
        
        # Set tensors for first model
        self.interpreter_1.set_tensor(self.input_details_1[1]['index'], self.states_1)
        self.interpreter_1.set_tensor(self.input_details_1[0]['index'], in_mag)
        
        # Run first model
        self.interpreter_1.invoke()
        
        # Get first model outputs
        out_mask = self.interpreter_1.get_tensor(self.output_details_1[0]['index'])
        self.states_1 = self.interpreter_1.get_tensor(self.output_details_1[1]['index'])
        
        # Apply mask and convert back to time domain
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        
        # Stage 2: Time domain processing
        # Reshape for second model
        estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
        
        # Set tensors for second model
        self.interpreter_2.set_tensor(self.input_details_2[1]['index'], self.states_2)
        self.interpreter_2.set_tensor(self.input_details_2[0]['index'], estimated_block)
        
        # Run second model
        self.interpreter_2.invoke()
        
        # Get enhanced output
        enhanced_block = self.interpreter_2.get_tensor(self.output_details_2[0]['index'])
        self.states_2 = self.interpreter_2.get_tensor(self.output_details_2[1]['index'])
        
        return np.squeeze(enhanced_block)
        
    def process_stream(self, indata, outdata, frames, time, status):
        """Real-time stream processing callback"""
        try:
            if status:
                print(f"Stream error: {status}")
            
            # Convert stereo to mono if needed by averaging channels
            if indata.shape[1] > 1:
                input_audio = np.mean(indata, axis=1)
            else:
                input_audio = np.squeeze(indata)
            
            # Update input buffer
            self.in_buffer[:-self.block_shift] = self.in_buffer[self.block_shift:]
            self.in_buffer[-self.block_shift:] = input_audio
            
            # Process current block
            enhanced = self._enhance_block(self.in_buffer)
            
            # Update output buffer
            self.out_buffer[:-self.block_shift] = self.out_buffer[self.block_shift:]
            self.out_buffer[-self.block_shift:] = np.zeros((self.block_shift))
            self.out_buffer += np.squeeze(enhanced)
            
            # Output processed audio (duplicate to all output channels if needed)
            if outdata.shape[1] > 1:
                outdata[:] = np.repeat(np.expand_dims(self.out_buffer[:self.block_shift], axis=1), outdata.shape[1], axis=1)
            else:
                outdata[:] = np.expand_dims(self.out_buffer[:self.block_shift], axis=-1)
            
        except Exception as e:
            print(f"Error in audio processing: {type(e).__name__}: {str(e)}")
            outdata.fill(0)
        
    def start_processing(self, device=None, latency=0.2):
        """
        Start real-time audio processing
        Args:
            device: Audio device ID or name
            latency: Stream latency in seconds
        """
        try:
            print("\n=== Starting Audio Processing ===")
            print(f"Sample rate: {self.sample_rate} Hz")
            print(f"Block length: {self.block_len_ms} ms ({self.block_len} samples)")
            print(f"Block shift: {self.block_shift_ms} ms ({self.block_shift} samples)")
            print(f"Latency: {latency*1000:.1f} ms")
            
            # Get device info
            if device is not None:
                device_info = sd.query_devices(device)
                print(f"\nDevice info: {device_info}")
                input_channels = device_info['max_input_channels']
                output_channels = device_info['max_output_channels']
                print(f"Input channels: {input_channels}, Output channels: {output_channels}")
                print(f"Device sample rate: {device_info['default_samplerate']} Hz")
                
                # Use device's native channel configuration
                if input_channels > 0 and output_channels > 0:
                    channels = (min(2, input_channels), min(2, output_channels))
                else:
                    channels = (1, 1)
            else:
                channels = (1, 1)
                print("\nNo device specified, using default")
                
            print(f"\nUsing channel configuration: {channels}")
            
            with sd.Stream(device=device,
                         samplerate=self.sample_rate, 
                         blocksize=self.block_shift,
                         dtype=np.float32, 
                         latency=latency,
                         channels=channels,  # Use detected channel configuration
                         callback=self.process_stream):
                print('\n' + '#' * 80)
                print('Noise reduction active - Press Return to quit')
                print('#' * 80)
                input()
        except KeyboardInterrupt:
            print("\nStopping noise reduction...")
        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {str(e)}")
            if device is not None:
                print("\nAvailable devices:")
                for i, dev in enumerate(sd.query_devices()):
                    print(f"\nDevice {i}:")
                    for key, value in dev.items():
                        print(f"  {key}: {value}") 