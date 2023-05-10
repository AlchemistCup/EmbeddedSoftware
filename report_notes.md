## Performance
- A C++ rewrite of the embedded software is unlikely to help too much with performance, as OpenCV just has a python wrapper meaning all of its library functions run in native C/C++. The performance difference is negligible (~4%) [cite](https://stackoverflow.com/a/13433330)
    - Interfacing with pi camera in C++ seems less straightforward (official docs recommend python library)

- Exploit efficient C++ implementation as much as possible by minimising use of python iteration (use numpy alternatives as much as possible for example)