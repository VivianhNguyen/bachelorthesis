#include <PDM.h>
#include <stdlib.h>

// buffer to read samples into, each sample is 16-bits
short sampleBuffer[256];
int bytesAvailable = 0;
unsigned int energy_array[256];

char read_index = 0;
char start_index = 0;

char frames_since_max = 12;

int threshold = 0;
int min_threshold = 0;

int min(int a, int b){
  if(a<b)
    return a;
  return b;
}

int max(int a, int b){
  if(a<b)
    return b;
  return a;
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

   // configure the data receive callback
  PDM.onReceive(onPDMdata);

  
  // initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate
  if (!PDM.begin(1, 11025)) {
    Serial.println("Failed to start PDM!");
    while (1);
  }

}

void loop() {
  // put your main code here, to run repeatedly:

}


void onPDMdata() {
  frames_since_max++;

  if(frames_since_max > 7)
    digitalWrite(LED_BUILTIN, LOW);

  // query the number of bytes available
  bytesAvailable = PDM.available();

  // read into the sample buffer
  PDM.read(sampleBuffer, bytesAvailable);

  energy_array[read_index] = 0;

  //sums the square values
  for(int i=0;i<256;++i)
    energy_array[read_index] += sampleBuffer[i] * sampleBuffer[i];

 
  int max_of_last_12 = 0;
  for(int i=1; i<13; ++i)
    max_of_last_12 = max(energy_array[start_index+i], max_of_last_12);

  if(frames_since_max >= 12 && energy_array[start_index] >= threshold && energy_array[start_index] >= max_of_last_12){
        frames_since_max = 0;
        threshold = max(min(energy_array[read_index], 1.25 * min_threshold), min_threshold);
        digitalWrite(LED_BUILTIN, HIGH);
      }
  
  read_index++;
  start_index++;
}
