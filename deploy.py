import sys
from cx_Freeze import setup, Executable

__author__      = "Tomáš Pazdiora"
__description__ = "Head Tracking Controller"
__app_name__    = "CamController"
__full_name__   = "Camera Controller"
__version__     = "1.0.0"
__guid__        = "{644245e2-c43e-4628-b4a3-bb6767ee6f98}"


base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name = __app_name__,
    version = __version__,
    description = __description__,
    author=__author__,
    executables = [
        Executable(
            "cam_controller.py",
            base=base,
            icon="app.ico",
            shortcutName=__full_name__,
            shortcutDir="DesktopFolder",
        )
    ],
    options = {
        'build_exe' : {
            'packages': ['numpy', 'cv2', 'keyboard', 'requests', 'simpleaudio'],
            'include_files': ['data/'],
            'optimize': 2,
            'include_msvcr': True,
        },
        'bdist_msi': {
            'install_icon': 'app.ico',
            'upgrade_code': __guid__,
            'add_to_path': False,
            'initial_target_dir': r'[ProgramFilesFolder]\{}'.format(__app_name__)
        }
    }
)