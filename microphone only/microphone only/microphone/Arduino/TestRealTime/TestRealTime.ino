#include <PDM.h>
#include <stdlib.h>

// buffer to read samples into, each sample is 16-bits
char sampleBuffer[512];
int bytesAvailable = 0;

void setup() {

  //the sample rate is 16Khz with a 16-bit depth that means 32KBytes/s are needed to fully transfer this signal
  //the baud rate represents the bit rate of the signal, 32*8 is 256Kbits/s, closest compatible baud rate of the nano is 500kbaud
  Serial.begin(500000);
  while (!Serial);

  // configure the data receive callback
  PDM.onReceive(onPDMdata);

  
  // optionally set the gain, defaults to 20
  PDM.setGain(95);

  // initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
}

void loop() {
  if(bytesAvailable){
    Serial.write(sampleBuffer, bytesAvailable);
    bytesAvailable = 0;
  }
}


void onPDMdata() {
  // query the number of bytes available
  bytesAvailable = PDM.available();

  // read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);
}