# TinyML Edge Anomaly Detection

Real-time vibration-based bearing fault detection on a Cortex-M4 microcontroller using an autoencoder trained on the CWRU bearing dataset.

## Research question
Can an autoencoder deployed on an STM32L4 (80 MHz, 128 KB RAM) detect incipient bearing faults within 100 ms of onset while consuming under 100 µJ per inference?

## Approach
1. Train a 1D-convolutional autoencoder on CWRU healthy data.
2. Use reconstruction error as anomaly score.
3. Quantize to INT8 with TFLite Micro.
4. Deploy on STM32L4 Nucleo board with an ADXL355 accelerometer.
5. Profile energy with an INA219 current sensor.

## Deliverables
- Training notebook in `src/python/`
- STM32 firmware in `src/firmware/`
- Energy + latency measurements in `results/`
