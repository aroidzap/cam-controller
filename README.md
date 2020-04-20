# [Camera Controller](https://github.com/aroidzap/cam-controller)
- 📷 webcamera 🤨 head tracking ⌨️ keyboard controller
- control your keyboard **arrow** keys with your 🕺 movements
- *press **spacebar** to reset tracking*
- *run app with `--wasd` switch to controll **WASD** keys instead*
- *run app with `--ijkl` switch to controll **IJKL** keys instead*
- *run app with `--no-top` switch to suppress always on top behavior (for Windows users)*
- *run app with `--buffer-frames` switch to buffer all frames from camera for tracking*
- *run app with `--tracker-downscale SCALE` switch to set tracking stage image downscaling (default is 4.0)*

-----------------------------------------
*app is using face recognition model from [https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)*

## Install Dependencies
- `sudo apt-get install python3-dev libasound2-dev -y` *(for Linux users)*
- `pip install -r requirements.txt`
- `pip install -r requirements-optional.txt` *(for Windows users)*

## Run
- `python cam_controller.py`
- open [![chrome-icon](https://www.google.com/chrome/static/images/favicons/favicon-16x16.png) **Chrome**](https://www.google.com/chrome/) and type into url bar: `chrome://dino`
- 🕹️ **enjoy** and **hit the highest score !!!** 🦖

-----------------------------------------
*you can **reset** `chrome://dino` **highest-score** by **clikcing it twice***

## Deploy
- `pip install -r requirements-deploy.txt`
- `python deploy.py bdist_msi` *(for Windows users)*
- `python deploy.py bdist_dmg` *(for Mac users)*
- `python deploy.py bdist_rpm` *(for Linux users)*
```
