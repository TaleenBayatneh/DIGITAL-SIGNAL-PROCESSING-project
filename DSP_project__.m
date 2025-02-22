%Load ECG Signal
load('105m.mat'); 
fs = 360; % Sampling frequency
ecg_signal = val(1, :); 
t = (0:length(ecg_signal)-1) / fs;
N = length(ecg_signal);

%Generate 50 Hz Power Line Interference
f_interf = 50; 
noiseAmplitude = 100; 
interference = noiseAmplitude * sin(2 * pi * f_interf * t);

noisySignal = ecg_signal + interference;

%Notch Filter Parameters
f_notch = 50;
wo = 2 * pi * f_notch / fs;
bw = wo / 15; 
r = 1 - bw;

bNotch = [1, -2*cos(wo), 1];
aNotch = [1, -2*r*cos(wo), r^2];

%Notch Filter
notchFilteredSignal = filter(bNotch, aNotch, noisySignal);

%RLS Filter Parameters
filter_order = 32; % Number of adaptive filter coefficients
lambda = 0.99; % 0.98 < Forgetting factor < 1
P = 1e4 * eye(filter_order); % Initial value for RLS covariance matrix
wRLS = zeros(filter_order, 1); % Initialize weights
RLSFilteredSignal = zeros(size(noisySignal)); % Initialize output

% RLS Filter Implementation
for n = filter_order:length(noisySignal)
    xRLS = noisySignal(n:-1:n-filter_order+1)'; 
    dRLS = ecg_signal(n);
    eRLS = dRLS - wRLS' * xRLS; 
    k = (P * xRLS) / (lambda + xRLS' * P * xRLS); 
    wRLS = wRLS + k * eRLS; 
    P = (P - k * xRLS' * P) / lambda; 
    RLSFilteredSignal(n) = wRLS' * xRLS; 
end

% --- NLMS Filter Parameters ---
mu = 0.01; % Step size
epsilon = 0.0001; % Regularization parameter
wNLMS = zeros(filter_order, 1); % Initialize weights
NLMSFilteredSignal = zeros(size(noisySignal));

% NLMS Filter Implementation
for n = filter_order:length(noisySignal)
    xNLMS = noisySignal(n:-1:n-filter_order+1)'; 
    dNLMS = ecg_signal(n);
    power = xNLMS' * xNLMS + epsilon; 
    e = dNLMS - wNLMS' * xNLMS; 
    wNLMS = wNLMS + (mu / power) * e * xNLMS;
    NLMSFilteredSignal(n) = wNLMS' * xNLMS; 
end

% --- Apply Manual Median Filter ---
WindowSize = 21; % Adjusted window size for ECG characteristics
manualFilteredSignal = manualMedianFilter(noisySignal, WindowSize);

% --- Evaluate Performance ---
[snrNotch, prdNotch, mseNotch] = evaluatePerformance(ecg_signal, notchFilteredSignal);
[snrRLS, prdRLS, mseRLS] = evaluatePerformance(ecg_signal, RLSFilteredSignal);
[snrNLMS, prdNLMS, mseNLMS] = evaluatePerformance(ecg_signal, NLMSFilteredSignal);

% --- Display Performance Metrics ---
fprintf('Performance Metrics:\n');
fprintf('Notch Filter: SNR = %.2f, PRD = %.2f%%, MSE = %.4f\n', snrNotch, prdNotch, mseNotch);
fprintf('RLS Filter: SNR = %.2f, PRD = %.2f%%, MSE = %.4f\n', snrRLS, prdRLS, mseRLS);
fprintf('NLMS Filter: SNR = %.2f, PRD = %.2f%%, MSE = %.4f\n', snrNLMS, prdNLMS, mseNLMS);

% --- Plot Results in Separate Figures ---
% Notch Filter Plot
figure;
plot(t, ecg_signal, 'r', 'DisplayName', 'Original ECG');
hold on;
plot(t, noisySignal, 'y', 'DisplayName', 'Noisy ECG');
plot(t, notchFilteredSignal, 'g', 'DisplayName', 'Notch Filtered ECG');
title('ECG Signal with Notch Filter');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
legend;

% RLS Filter Plot
figure;
plot(t, ecg_signal, 'r', 'DisplayName', 'Original ECG');
hold on;
plot(t, noisySignal, 'y', 'DisplayName', 'Noisy ECG');
plot(t, RLSFilteredSignal, 'g', 'DisplayName', 'RLS Filtered ECG');
title('ECG Signal with RLS Filter');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
legend;

% NLMS Filter Plot
figure;
plot(t, ecg_signal, 'r', 'DisplayName', 'Original ECG');
hold on;
plot(t, noisySignal, 'y', 'DisplayName', 'Noisy ECG');
plot(t, NLMSFilteredSignal, 'g', 'DisplayName', 'NLMS Filtered ECG');
title('ECG Signal with NLMS Filter');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
legend;

% original signal Plot
figure;
plot(t, ecg_signal, 'r', 'DisplayName', 'Original ECG');
hold on;
title('original signal');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
legend;

% median plot
figure;
plot(t, ecg_signal, 'r', 'DisplayName', 'Original ECG'); % Original ECG
hold on;
plot(t, noisySignal, 'y', 'DisplayName', 'Noisy ECG'); % Noisy ECG
plot(t, manualFilteredSignal, 'g', 'DisplayName', 'Median Filtered ECG'); % Median Filtered ECG
title('ECG Signal with Median Filter');
xlabel('Time (s)');
ylabel('Amplitude (mV)');
legend;

% Frequency Domain Representation
f = linspace(0, fs/2, N/2);
noisySpectrum = abs(fft(noisySignal));
notchSpectrum = abs(fft(notchFilteredSignal));
RLSSpectrum = abs(fft(RLSFilteredSignal));
NLMSSpectrum = abs(fft(NLMSFilteredSignal));

figure;
subplot(2, 2, 1); plot(f, noisySpectrum(1:N/2)); title('Noisy Signal Spectrum'); xlabel('Frequency (Hz)'); ylabel('Magnitude');
subplot(2, 2, 2); plot(f, notchSpectrum(1:N/2)); title('Notch Filter Spectrum'); xlabel('Frequency (Hz)'); ylabel('Magnitude');
subplot(2, 2, 3); plot(f, RLSSpectrum(1:N/2)); title('RLS Filter Spectrum'); xlabel('Frequency (Hz)'); ylabel('Magnitude');
subplot(2, 2, 4); plot(f, NLMSSpectrum(1:N/2)); title('NLMS Filter Spectrum'); xlabel('Frequency (Hz)'); ylabel('Magnitude');

% Power Spectral Density (PSD)
figure;
subplot(2, 2, 1); pwelch(noisySignal, [], [], [], fs); title('Noisy Signal PSD');
subplot(2, 2, 2); pwelch(notchFilteredSignal, [], [], [], fs); title('Notch Filter PSD');
subplot(2, 2, 3); pwelch(RLSFilteredSignal, [], [], [], fs); title('RLS Filter PSD');
subplot(2, 2, 4); pwelch(NLMSFilteredSignal, [], [], [], fs); title('NLMS Filter PSD');

% Spectrogram Representation
figure;
subplot(2, 2, 1); spectrogram(noisySignal, 128, 120, 128, fs, 'yaxis'); title('Noisy Signal Spectrogram');
subplot(2, 2, 2); spectrogram(notchFilteredSignal, 128, 120, 128, fs, 'yaxis'); title('Notch Filter Spectrogram');
subplot(2, 2, 3); spectrogram(RLSFilteredSignal, 128, 120, 128, fs, 'yaxis'); title('RLS Filter Spectrogram');
subplot(2, 2, 4); spectrogram(NLMSFilteredSignal, 128, 120, 128, fs, 'yaxis'); title('NLMS Filter Spectrogram');

% Plot results in Time Domain
figure;
subplot(5, 1, 1); plot(t, ecg_signal); title('Original ECG Signal');
subplot(5, 1, 2); plot(t, noisySignal); title('Noisy ECG Signal');
subplot(5, 1, 3); plot(t, notchFilteredSignal); title('Notch Filtered Signal');
subplot(5, 1, 4); plot(t, RLSFilteredSignal); title('RLS Filtered Signal');
subplot(5, 1, 5); plot(t, NLMSFilteredSignal); title('Adaptive NLMS Filtered Signal');




% --- Function Definitions ---
function filteredSignal = manualMedianFilter(noisySignal, windowSize)
    % Apply a manual median filter to the noisy signal
    filteredSignal = zeros(size(noisySignal));
    halfWindow = floor(windowSize / 2);
    for i = 1:length(noisySignal)
        startIdx = max(i - halfWindow, 1);
        endIdx = min(i + halfWindow, length(noisySignal));
        filteredSignal(i) = median(noisySignal(startIdx:endIdx));
    end
end

function [snr, prd, mse] = evaluatePerformance(original, filtered)
    % Evaluate performance metrics for signal filtering
    mse = mean((original - filtered).^2);
    snr = 10 * log10(sum(original.^2) / sum((original - filtered).^2));
    prd = 100 * sqrt(sum((original - filtered).^2) / sum(original.^2));
end
