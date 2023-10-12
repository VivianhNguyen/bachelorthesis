#include <PDM.h>
#include <stdlib.h>

// buffer to read samples into, each sample is 16-bits
char sampleBuffer[512];

//measure the time it takes to gather the signal, for testing purposes.
unsigned long currentTimeStart;
unsigned long currentTimeEnd;

// array of pointers, here all the audio data will be saved into ram.
// 192 pointers to arrays of size 512 bytes, equal to 3 seconds of audio
// enough for the purposes of testing
char audio[98816];
long int totalBytes = 0;
bool ok = false;

void setup() {
  //offers time to upload code and run python script
  delay(5000);
  //the sample rate is 16Khz with a 16-bit depth that means 32KBytes/s are needed to fully transfer this signal
  //the baud rate represents the bit rate of the signal, 32*8 is 256Kbits/s, closest compatible baud rate of the nano is 500kbaud
  Serial.begin(500000);
  while (!Serial);

  // configure the data receive callback
  PDM.onReceive(onPDMdata);

  currentTimeStart = millis();
  
  // optionally set the gain, defaults to 20
  // PDM.setGain(30);

  // initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }
}

void loop() {
  if(totalBytes > 98303 && ok == false)
  {  
    ok = true;
    Serial.write(audio, totalBytes);
  }
      
}

void onPDMdata() {
  // query the number of bytes available
  int bytesAvailable = PDM.available();

  // read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);


  if(totalBytes < 98304)
  {
    currentTimeEnd = millis();
    memcpy(audio+totalBytes, sampleBuffer, bytesAvailable);
    totalBytes += bytesAvailable;
  }

}