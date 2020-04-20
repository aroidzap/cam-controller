__author__      = "Tomáš Pazdiora"
__description__ = "Head Tracking Controller"
__app_name__    = "CamController"
__full_name__   = "Camera Controller"
__version__     = "1.1.0"
__app_main__    = "cam_controller.py"
__app_icon__    = "app.ico"
__app_data__    = ["data", "README.md", "LICENSE"]
__guid__        = "{644245e2-c43e-4628-b4a3-bb6767ee6f98}"

import os
import platform
import subprocess
from cx_Freeze import setup, Executable

# Get third-party licenses
if not os.path.exists('build'):
    os.makedirs('build')
subprocess.call(['pip-licenses', '--with-license-file', '--format=markdown', '--no-license-path', '--output-file=build/THIRDPARTY_LICENSES'])

# PyInstaller
class PyInstallerExecutable(Executable):
    def __init__(self, script, initScript = None, base = None,
        targetName = None, icon = None, shortcutName = None,
        shortcutDir = None, copyright = None, trademarks = None):

        subprocess.call(['pyinstaller', 
            '--specpath=build',
            '--clean', '--onedir', 
            '--noconsole', '--windowed',
            '--icon=../{}'.format(icon),
            '--name={}'.format(targetName),
            script], shell=True)
        
        if not os.path.exists('build'):
            os.makedirs('build')

        ext = ""
        if platform.system().lower().startswith('windows'):
            ext = ".exe"
        elif platform.system().lower().startswith('darwin'):
            ext = ".app"

        with open('build/launcher.py', 'w') as file:
            file.write('import subprocess; subprocess.call(["{}{}"], shell=True)'.format(targetName, ext))

        if base is None:
            base = "Win32GUI" if platform.system().lower().startswith('windows') else None

        super().__init__('build/launcher.py', initScript, base, "launcher", 
            icon, shortcutName, shortcutDir, copyright, trademarks)

# cx_Freeze setup
setup(
    name = __app_name__,
    version = __version__,
    description = __description__,
    author=__author__,
    executables = [
        PyInstallerExecutable(
            __app_main__,
            icon = __app_icon__,
            targetName = __app_name__,
            shortcutName = __full_name__,
            shortcutDir = "DesktopFolder",
        )
    ],
    options = {
        'build_exe' : {
            'include_files' : [
                ('dist/{}/'.format(__app_name__), '.'), 
                ('build/THIRDPARTY_LICENSES', 'THIRDPARTY_LICENSES')] + 
                [(file, '') for file in __app_data__],
            'include_msvcr' : True
        },
        'bdist_msi': {
            'install_icon': __app_icon__,
            'upgrade_code': __guid__,
            'add_to_path': False,
            'initial_target_dir': r'[ProgramFilesFolder]\{}'.format(__app_name__),
            'data': {"Shortcut": [(
                'DesktopShortcut','DesktopFolder', __full_name__, 
                'TARGETDIR', '[TARGETDIR]launcher.exe', None, 
                __description__, None, None, None, None, 'TARGETDIR')]}
        },
        'bdist_mac': {
            'iconfile': __app_icon__,
            'bundle_name': __app_name__
        },
        'bdist_dmg': {
            'applications_shortcut': True
        },
        'bdist_rpm': {
            'icon': __app_icon__,
            'requires': ['python3-dev', 'libasound2-dev']
        }
    }
)