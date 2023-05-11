## Performance
- A C++ rewrite of the embedded software is unlikely to help too much with performance, as OpenCV just has a python wrapper meaning all of its library functions run in native C/C++. The performance difference is negligible (~4%) [cite](https://stackoverflow.com/a/13433330)
    - Interfacing with pi camera in C++ seems less straightforward (official docs recommend python library)

- Exploit efficient C++ implementation as much as possible by minimising use of python iteration (use numpy alternatives as much as possible for example)

- Raspberry Pi 4B is ~3x more powerful than 3B+ (current model) + can have more RAM (would recommend 4GB for image processing) => much faster FPS

### Improvements
- Sped up grayscale conversion using different image format (taking advantage of specialised hardware) => 2.4x performance boost - (see [here](https://github.com/raspberrypi/picamera2/issues/698))
- Updated board square segmentation to use numpy operations rather than python for loops (need to actually test if this is faster)

