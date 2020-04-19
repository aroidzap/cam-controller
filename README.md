# About
- Camera Controller is head tracking controller, which simulates arrow key presses

# Run
- `pip install -r requirements.txt`
- `python cam_controller.py`
- open **Chrome** and type into url bar: `chrome://dino`
- enjoy !!!

# Requirements
```
pip install numpy
pip install opencv-contrib-python
pip install keyboard
pip install requests
pip install simpleaudio
```

# Deploy
```
pip install pyinstaller
pip install --upgrade setuptools<45.0.0

pyinstaller --clean --onedir --noconsole --add-data="data/*;data" --icon=app.ico --name CamController cam_controller.py
```