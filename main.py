import numpy as np
import pyaudio
from scipy.signal import resample

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

p = pyaudio.PyAudio();

def find_device(name):
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if name in device_info["name"]:
            return i
    return None

vb_index = find_device('CABLE Input')

if vb_index is None:
    raise ValueError("VB-Audio virtual cable not found")


#input stream
input_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

#output stream
output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK, output_device_index=vb_index)

pitch_factor = 1.2

def pitch_shift(data, pitch_factor):
    audio_data = np.frombuffer(data, dtype=np.int16)

    factor = int(len(audio_data) / pitch_factor)
    resampled_audio = resample(audio_data, factor)

    return resampled_audio.astype(np.int16)


try:
    while True:

        data = input_stream.read(CHUNK)

        shifted_audio = pitch_shift(data, pitch_factor)

        output_stream.write(shifted_audio.tobytes())
except KeyboardInterrupt:
    print("Exiting")


input_stream.stop_stream()
input_stream.close
output_stream.stop_stream()
output_stream.close()
p.terminate();
