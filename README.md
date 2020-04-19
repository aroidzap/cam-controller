# Camera Controller 
- 📷 webcamera 🤨 head tracking ⌨️ keyboard controller
- control your keyboard **arrow** keys with your 🕺 movements
- *press **spacebar** to reset tracking*
- *run app with `--wasd` switch to controll **WASD** keys instead*
- *run app with `--ijkl` switch to controll **IJKL** keys instead*
- *run app with `--no-top` switch to suppress always on top behavior (for Windows users)*

-----------------------------------------
*app is using face recognition model from [https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)*

## Run
- `pip install -r requirements.txt`
- `pip install -r requirements-optional.txt` *(for Windows users)*
- `python cam_controller.py`
- open [![chrome-icon](https://www.google.com/chrome/static/images/favicons/favicon-16x16.png) **Chrome**](https://www.google.com/chrome/) and type into url bar: `chrome://dino`
- 🕹️ **enjoy** and **hit the highest score !!!** 🦖

-----------------------------------------
*you can **reset** `chrome://dino` **highest-score** by **clikcing it twice***

## Requirements
```
pip install numpy
pip install opencv-contrib-python
pip install keyboard
pip install requests
pip install simpleaudio
```
optional: *(for Windows users)*
```
pip install pywin32
```

## Deploy
### Using cx_Freeze
```
pip install cx_Freeze
python deploy.py bdist_msi
```
### Using PyInstaller
```
pip install pyinstaller
pip install --upgrade setuptools<45.0.0
pyinstaller --clean --onedir --noconsole --add-data="data/*;data" --icon=app.ico --name CamController cam_controller.py
```
