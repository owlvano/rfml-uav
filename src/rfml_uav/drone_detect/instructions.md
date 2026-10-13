## Abstract

The DroneDetect dataset consists of 7 different models of popular Unmanned Aerial Systems (UAS) including the new DJI Mavic 2 Air S, DJI Mavic Pro, DJI Mavic Pro 2, DJI Inspire 2, DJI Mavic Mini, DJI Phantom 4 and the Parrot Disco. Recordings were collected using a Nuand BladeRF SDR and using open source software GNURadio. There are 4 subsets of data included in this dataset, the UAS signals in the presence of Bluetooth interference, in the presence of Wi-Fi signals, in the presence of both and with no interference. 3 flight modes are captured - switched on, hovering and flying.

Instructions:
- Sample rate: 60Mbits/s
- Bandwidth: 28MHz
- Centre Freq: 2.4375GHz

Each recording consists of 1.2 x 10^8 complex samples equating to 2 seconds recording time. Data is saved into ‘.dat’ files  and the complex data is saved as interleaved floats. ‘load_data.py’ is included for the data to be loaded into python and further split into smaller samples 20ms in length.

Files are categorised by interference, then by flight mode:

- Switched on = ON
- Hovering = HO
- Flying = FY

Each file name uses an interference identifier, 00 for a clean signal, 01 for Bluetooth only, 10 for Wi-Fi only and 11 for Bluetooth and Wi-Fi interference concurrently. An example file name for Mavic Mini switched on in the presence of Bluetooth and Wi-Fi interference would be:
> MIN + 11 + 00 + 00 = MIN_1100_00.dat