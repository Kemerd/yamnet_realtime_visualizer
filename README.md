# Realtime YAMNET Sound Event Detection 🎤 🔊

A real-time Sound Event Detection (SED) system powered by [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), utilizing your device's microphone for continuous audio analysis, with support for noise reduction and input gain preprocessing.

![Screenshot of the application in action](./Screenshot.png)

## Overview 🎯

This application performs real-time sound event detection using Google's YAMNet deep neural network model, which can identify 521 different audio events. The system processes audio input every second and displays the top 5 most probable sound events it detects.

## Features ✨

- 🎧 Real-time audio processing using PyAudio
- 🤖 Continuous sound event detection using YAMNet
- 📊 Display of top 5 detected sound events with confidence scores
- 🎵 Support for 521 different audio event classes
- ⚡ Low-latency processing (1-second intervals)
- 🔇 Noise reduction capabilities using DTLN
- 🎚️ Adjustable input gain control

## Prerequisites 🛠️

- Python 3.6 or higher
- PyAudio
- TensorFlow 2.x
- NumPy

## Installation 📥

1. Clone this repository:
```bash
git clone https://github.com/yourusername/realtime_YAMNET.git
cd realtime_YAMNET
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

Or install packages manually:
```bash
pip install tensorflow>=2.4.0
pip install numpy>=1.19.2
pip install pyaudio>=0.2.11
pip install sounddevice>=0.4.1
pip install scipy>=1.6.0
pip install librosa>=0.8.0
pip install resampy>=0.2.2
pip install matplotlib>=3.3.3
pip install tqdm>=4.54.1
pip install pandas>=1.2.0
```

4. Download the pre-trained YAMNet model:
```bash
# Download manually from Google Storage
wget https://storage.googleapis.com/audioset/yamnet.h5
# Place the downloaded file in the yamnet folder
mv yamnet.h5 yamnet/
```

## Usage 🚀

Run the application with:
```bash
python realtime_YAMNET.py
```

Optional command-line arguments:
```bash
python realtime_YAMNET.py --noise_reduction True --input_gain 1.5
```

The system will immediately begin capturing audio from your default microphone and display real-time detection results.

## How It Works 🔍

1. The system captures audio input through your device's microphone using PyAudio
2. Audio is processed in 1-second segments
3. Optional noise reduction is applied using DTLN
4. Input gain adjustment is applied if specified
5. YAMNet analyzes the audio segment and produces probability scores for 521 different sound events
6. The top 5 most probable sound events are displayed in real-time
7. The process continues until the application is terminated

## Supported Sound Events 🔉

YAMNet can detect 521 different audio events from the [AudioSet ontology](https://research.google.com/audioset/), including:
- 🗣️ Human sounds (speech, whistling, laughing)
- 🐾 Animal sounds (dog barking, cat meowing, bird songs)
- 🎸 Musical instruments
- 🚗 Vehicle sounds
- 🌳 Environmental sounds
- And many more...

## Troubleshooting 🔧

Common issues and solutions:

1. **PyAudio installation fails**:
   - On Ubuntu/Debian: `sudo apt-get install python3-pyaudio`
   - On Windows: `pip install pipwin && pipwin install pyaudio`
   - On macOS: `brew install portaudio && pip install pyaudio`

2. **CUDA/GPU issues**:
   - Ensure CUDA toolkit and cuDNN are properly installed
   - Check TensorFlow GPU support with `tf.test.is_built_with_cuda()`

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) by Google Research
- [AudioSet](https://research.google.com/audioset/) project
- [realtime_YAMNET](https://github.com/SangwonSUH/realtime_YAMNET) original fork
- [DTLN](https://github.com/breizhn/DTLN) for noise reduction

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
