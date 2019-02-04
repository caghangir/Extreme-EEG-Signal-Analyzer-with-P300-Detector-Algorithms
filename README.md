# Extreme-EEG-Signal-Analyzer-with-P300-Detector-Algorithms

I created Extreme EEG Signal Analyzer for my research projects at first. However, since I have been dealing with all stages of Brain-computer Interface based projects, I collected all relevant algorithms in all stages of creating BCI systems.

Basically, this Python code covers these fields;
* Pre-processing of EEG signals (noise filters, subsampling, notch filter, artifact reduction, bandpass/low-high pass filters etc.)
* Spectral analyses of EEG signals included fourier transforms, discrete wavelet transforms, extraction of energy of frequency bins which are shown below;
     - delta (0.5 - 3 Hz)
     - theta (3 - 8 Hz)
     - low alpha (8 - 10 Hz)
     - high alpha (10 - 12 Hz)
     - low beta (12 - 30 Hz)
     - high beta (30 - 38 Hz)
     - low gamma (38 - 40 Hz)
     - high gamma (+40 Hz)
     
* Other Spectral Features;
  - Hurst exponent
  - Detrended Fluctuation Analysis
  - Hjorth mobility and complexity of a time series
  - Hjorth Fractal Dimension of a time series
  - Petrosian Fractal Dimension
  
* Mel-spectrogram
* P300 Detection Algorithms
* Plot Variety of P300
* Construction of P300 predictor machine learning models using SKlearn.
* Confusion Matrix creation and plot.


I will post detailed instructions soon.
