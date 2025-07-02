/*
  ============================================================================
  Project Title   : Multichannel Analog Acquisition (18 Channels)
  Board           : Teensy 4.1
  Serial Port     : COMX (Replace "X" with the appropriate port number)
  Keyboard Layout : AZERTY (French)
  CPU Frequency   : 600 MHz
  Optimization    : Faster (Performance optimized)
  USB Interface   : Serial (for data streaming to MATLAB)
  Author          : Madou FALL
  Date            : June 20, 2025
  ----------------------------------------------------------------------------
  Description     : 
    This program reads analog signals from 18 input channels using the 
    Teensy 4.1 board and streams the data in real time over a serial 
    connection. The output format is CSV (Comma-Separated Values) and 
    is designed to be directly imported into MATLAB for processing and 
    visualization.

  Sampling Rate   : 200 Hz (each channel)
  Data Format     : CSV (1 row = 18 comma-separated samples per timestamp)
  Buffer Size     : 10,000 samples per channel (circular buffer)
  
  Key Features    :
    - High-speed analog data acquisition
    - Configurable ADC resolution and speed
    - Compatible with MATLAB post-processing
    - Lightweight, real-time streaming via USB Serial

  Dependencies    :
    - ADC by pedvide (Teensy-compatible ADC library)
    - SPI.h (included by default)

  Notes           :
    - Ensure the Teensy 4.1 is connected to the correct COM port.
    - Set the correct sampling rate in MATLAB to match 200 Hz.
    - Match physical wiring to the 18 analog pins defined in the code.

  ============================================================================
*/

#include <ADC.h>         // Teensy ADC library
#include <ADC_util.h>    // ADC utility functions
#include <SPI.h>         // Not used here but often needed for peripherals

#define nChannels 18     // Number of analog channels to acquire
#define nSamples 10000   // Buffer size for each channel
#define RESOLUTION 10    // ADC resolution in bits (0-1023)
#define AVERAGING 1      // No hardware averaging
#define SAMPRATE 200     // Target sample rate (Hz)

// Allocate data buffer for all channels in DMA-capable memory
DMAMEM uint16_t data[nChannels][nSamples];

// Create ADC controller object
ADC *adc = new ADC();

// Circular buffer write index
uint16_t writeIndex = 0;

// Define the 18 analog input pins used
uint8_t adc_pins[nChannels] = {
  A0, A1, A2, A3, A4, A5,
  A6, A7, A8, A9, A10, A11,
  A12, A13, A14, A15, A16, A17
};

void setup() {
  // Start serial communication for data transfer (to MATLAB)
  Serial.begin(250000);
  while (!Serial);  // Wait until the Serial port is ready

  // Configure ADC settings for fast, precise sampling
  adc->adc0->setAveraging(AVERAGING);
  adc->adc0->setResolution(RESOLUTION);
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);

  Serial.println("Acquisition en cours...");  // Acquisition started
}

void loop() {
  // Step 1: Read all 18 channels
  for (int ch = 0; ch < nChannels; ch++) {
    uint16_t val = adc->analogRead(adc_pins[ch]);
    data[ch][writeIndex] = val;  // Store in buffer
  }

  // Step 2: Send one row of data to MATLAB (comma-separated)
  for (int ch = 0; ch < nChannels; ch++) {
    Serial.print(data[ch][writeIndex]);
    if (ch < nChannels - 1)
      Serial.print(",");  // Separate values with commas
  }
  Serial.println();  // End the row

  // Step 3: Update write index (circular buffer)
  writeIndex = (writeIndex + 1) % nSamples;

  // Step 4: Delay to match 200 Hz sample rate
  delay(1000 / SAMPRATE);
}

