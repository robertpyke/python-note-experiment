from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from tempfile import mktemp
import os
import logging
import sys
import numpy as np
from scipy.signal import butter, lfilter
import math


def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# explicit function to convert
# edge frequencies
def convertX(f_sample, f):
    w = []
      
    for i in range(len(f)):
        b = 2*((f[i]/2)/(f_sample/2))
        w.append(b)
  
    omega_mine = []
  
    for i in range(len(w)):
        c = (2/Td)*np.tan(w[i]/2)
        omega_mine.append(c)
  
    return omega_mine

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

logger = logging.getLogger(__name__)

audio_file_path = os.path.join(os.path.dirname(__file__), "keyboard_chords.mp3")
#audio_file_path = os.path.join(os.path.dirname(__file__), "basic_piano.mp3")

# audio_file_path = os.path.join(os.path.dirname(__file__), "peaches_trimmed.mp3")

mp3_audio = AudioSegment.from_file(audio_file_path)  # read mp3
mp3_audio.set_channels(1) # set it to mono
wname = mktemp(".wav")  # use temporary file
mp3_audio.export(wname, format="wav", parameters=["-ac", "1"])  # convert to wav

#plt.specgram(data, Fs=FS, NFFT=128, noverlap=0)  # plot
#plt.show()

sample_rate, samples = wavfile.read(wname)
logger.info("Sample Rate: %s", sample_rate)



# Specifications of Filter
  
# sampling frequency
f_sample = sample_rate
  


tolerance = 5
stop_band = 20
# pass band frequency
f_pass = [261.626-tolerance, 261.626+tolerance]
# pass band frequency
f_stop = [261.626-tolerance-stop_band, 261.626+tolerance+stop_band]
  
# pass band ripple
fs = 0.5  
# Sampling Time
Td = 1
# pass band ripple
g_pass = 0.4
# stop band attenuation
g_stop = 50

# Conversion to prewrapped analog 
# frequency 
omega_p=convertX(f_sample,f_pass)
omega_s=convertX(f_sample,f_stop)
    
# Design of Filter using signal.buttord 
# function 
N, Wn = signal.buttord(omega_p, omega_s, 
                       g_pass, g_stop, 
                       analog=True) 
    
    
# Printing the values of order & cut-off frequency
# N is the order 
print("Order of the Filter=", N) 
  
# Wn is the cut-off freq of the filter 
print("Cut-off frequency= {:} rad/s ".format(Wn)) 


plt.specgram(samples, Fs=sample_rate, NFFT=128, noverlap=0)  # plot
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
#plt.show()

f, t, Sxx = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
ax = plt.gca()
ax.set_ylim([0, 600])
#plt.show()





tolerance = 5

# Filter a noisy signal.
c_filetered_samples = butter_bandpass_filter(samples, 261.626-tolerance, 261.626+tolerance, sample_rate)
d_filetered_samples = butter_bandpass_filter(samples, 293.665-tolerance, 293.665+tolerance, sample_rate)
e_filetered_samples = butter_bandpass_filter(samples, 329.628-tolerance, 329.628+tolerance, sample_rate)
f_filetered_samples = butter_bandpass_filter(samples, 349.228-tolerance, 349.228+tolerance, sample_rate)
g_filetered_samples = butter_bandpass_filter(samples, 391.995-tolerance, 391.995+tolerance, sample_rate)
e5_filetered_samples = butter_bandpass_filter(samples, 659.255-tolerance, 659.255+tolerance, sample_rate)


logger.info("c_filetered_samples SHAPE %s",  c_filetered_samples.shape)

samples_to_plot = [
    ("ORIGINAL", samples),
    ("C4", c_filetered_samples),
    ("D4", d_filetered_samples),
    ("E4", e_filetered_samples),
    ("F4", f_filetered_samples),
    ("G4", g_filetered_samples),
]

vmin = min(samples)
vmax = max(samples)
figure, axis = plt.subplots(len(samples_to_plot), 1)

for i, val in enumerate(samples_to_plot):
    logger.info("val: %s", val)
    title, i_samples = val
    j_samples = i_samples
    #j_samples = [abs(x) for x in i_samples]
    #j_samples = [0 if x < 1_000 else 1 for x in i_samples]
    
    #plt.plot(j_samples)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()

    logger.info("Plotting %s", title)
    
    i_max = max(i_samples)
    logger.info("Relative max %d:%d", i_max, vmax)

    logger.info("SHAPE %s", i_samples.shape)

    # sample rate = 44100
    # shape (number of samples) = 354816
    # shape/sample-sate = 8.0457... (seconds of the source)
    logger.info("instance SHAPE %s",  i_samples[0].shape)
    logger.info("instance %s",  i_samples[0])
     
    f, t, Sxx = signal.spectrogram(i_samples, sample_rate)

    logger.info("f SHAPE %s",  f.shape)
    logger.info("t SHAPE %s",  t.shape)
    logger.info("Sxx SHAPE %s",  Sxx.shape)
    
    # f SHAPE (129,)
    # t SHAPE (1583,)
    # Sxx SHAPE (129, 1583)
    axis[i].plot(j_samples)
    
    
    # axis[i].pcolormesh(t, f, Sxx, shading='nearest')
    axis[i].set_title(title)
    axis[i].set_ylabel(f'Frequency [Hz]')
    axis[i].set_xlabel('Time [sec]')
    axis[i].figure.set_figheight(20)
    axis[i].figure.set_figwidth(20)
    axis[i].set_ylim([vmin, vmax])
    
plt.show()
