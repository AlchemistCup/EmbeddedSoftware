import RPi.GPIO as gpio
from smbus2 import SMBus
import cv2
from picamera2 import Picamera2
import time

MUX_INFO = {
    'i2c_address': 0x70,
    'i2c_offset': 0x00,
    'A': {
        'i2c_data': 0x04,
        'gpio_sel': [0,0,1]
    },
    'B': {
        'i2c_data': 0x05,
        'gpio_sel': [1,0,1]
    },
    'C': {
        'i2c_data': 0x06,
        'gpio_sel': [0,1,0]
    },
    'D': {
        'i2c_data': 0x07,
        'gpio_sel': [1,1,0]
    }
}

def main():
    setup_gpio()
    with SMBus(1) as i2c:
        start = time.time()
        for cam in ['A', 'B', 'C', 'D']:
            print(f"Testing camera {cam}")
            i2c.write_byte_data(MUX_INFO['i2c_address'], MUX_INFO['i2c_offset'], MUX_INFO[cam]['i2c_data'])
            set_gpio(MUX_INFO[cam]['gpio_sel'])   
            picam = setup_camera()         
            img = picam.capture_array()
            img = img[:720, :1280]
            cv2.imwrite(f'camera_{cam}.jpg', img)
            picam.close()
        end = time.time()
        print(f'Took {end - start:.2f} seconds')

def setup_gpio():
    gpio.setwarnings(False)
    gpio.setmode(gpio.BOARD)

    gpio.setup(7, gpio.OUT)
    gpio.setup(11, gpio.OUT)
    gpio.setup(12, gpio.OUT)

def set_gpio(data):
    assert len(data) == 3
    for pin, val in zip([7, 11, 12], data):
        gpio.output(pin, val)

def setup_camera() -> Picamera2:
    picam = Picamera2()

    config = picam.create_preview_configuration() # Test if still config are better
    config['main']['size'] = (1280, 720)
    config['main']['format'] = 'YUV420'
    picam.align_configuration(config)
    picam.configure(config)
    picam.start()

    return picam

if __name__ == "__main__":
    main()
