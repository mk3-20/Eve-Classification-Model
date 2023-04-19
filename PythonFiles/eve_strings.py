import os
from pathlib import Path

CWD = os.getcwd()
PARENT_DIR_BACK_SLASH = Path(CWD).parent
PARENT_DIR = PARENT_DIR_BACK_SLASH.as_posix()
MAIN_WINDOW_TITLE = 'Eve - Classification Model'

ICON_QRC_PATH = ':/img/icon_img'

EVE_GIF_PATH = "eve_trial_gif.gif"