# Requirements
```
pip install numpy
pip install opencv-contrib-python
pip install keyboard
pip install requests
```

# Deploy
```
pip install pyinstaller
pip install --upgrade setuptools<45.0.0

pyinstaller --clean --onefile --noconsole --icon=app.ico --name CamController cam_controller.py
```

# Run
- `python cam_controller.py`
- open **Chrome** and type into url bar: `chrome://dino`
- enjoy !!!