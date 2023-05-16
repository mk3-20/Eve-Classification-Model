import os
from pathlib import Path
from enum import Enum
from random import randint

CWD = os.getcwd()
PARENT_DIR_BACK_SLASH = Path(CWD).parent
PARENT_DIR = PARENT_DIR_BACK_SLASH.as_posix()
MAIN_WINDOW_TITLE = 'Eve - Classification Model'
PREFERENCES_PATH = "save_file.json"
QUESTION_FILE_PATH = "ask_eve_qa.json"
KEY_IMAGE_FOLDER = "ImageFolder"
KEY_SAVE_FOLDER = "SaveFolder"

UI_FOLDER_PATH = CWD + "\\UI\\"


DEFAULT_ANIM_INPUT_IMAGE_VERTICAL = 500
DEFAULT_ANIM_INPUT_IMAGE_HORIZONTAL = 1000

DEFAULT_DELAY_INTRO = 3000
DEFAULT_DELAY_ACCURACY = 1500
DEFAULT_DELAY_CLASSIFICATION = 1500
DEFAULT_DELAY_THINKING = 500
DEFAULT_FRAME_EXPAND_TIMING = 1000
DEFAULT_FRAME_COLLAPSE_TIMING = 500
DEFAULT_RESULTS_EXPAND_TIMING = 500
DEFAULT_RESULTS_COLLAPSE_TIMING = 1000

COLLAPSE_LEFT_ICON = UI_FOLDER_PATH + "left_icon.png"
COLLAPSE_RIGHT_ICON = UI_FOLDER_PATH + "right_icon.png"
RESET_ICON = UI_FOLDER_PATH + "reset_icon.png"
IMPORT_ICON = UI_FOLDER_PATH + "import_icon.png"
IMAGE_ICON = UI_FOLDER_PATH + "image_icon.png"
SPEED_UP_ICON = UI_FOLDER_PATH + "speed_up_icon.png"
SLOW_DOWN_ICON = UI_FOLDER_PATH + "slow_down_icon.png"
DROP_DOWN_ICON = UI_FOLDER_PATH + "drop_down_arrow.png"
DROP_UP_ICON = UI_FOLDER_PATH + "drop_up_arrow.png"

EVE_POSES_PATH = UI_FOLDER_PATH + "eve_poses"
CAT_BASKET_IMG = UI_FOLDER_PATH + "cat_basket_icon.jpg"
DOG_BASKET_IMG = UI_FOLDER_PATH + "dog_basket_icon.jpg"


EVE_CELEBRATING = "eve_celebrating.png"
EVE_HANDBAG = "eve_handbag.png"

EVE_EXPLAINING = "eve_explaining.png"
EVE_HANDBOARD = "eve_handboard.png"
EVE_HANDFOLD = "eve_handfold.png"
EVE_POINTING = "eve_pointing.png"
EVE_THINKING = "eve_thinking.png"

THINKING_POSES = [EVE_THINKING, EVE_HANDBOARD, EVE_HANDFOLD]
POINTING_POSES = [EVE_POINTING, EVE_EXPLAINING]

DIALOGUE_THINKING_LIST = ["I think that's...", "Ooo this one is surely..", "Hmmmm..", "Awww look at that..",
                          "Gotta think hard for this one.. "]
DIALOGUE_DOG_LIST = ["A cute lil doggo", "A dog", "A good doggy", "A DOG!!", "A Dawg", "doggy :))"]
DIALOGUE_CAT_LIST = ["A cat", "A Fluffy Kitty", "an El Gato", "A CAT!!", "kitty cat :D"]

ANSWER_HEIGHT = 401

SS_QUES_OFF = """
QPushButton {
	border-color: rgba(0, 0 ,0,0.6);
	border-radius: 10px;
	background-color: rgb(255, 229, 255);
	text-align:left;
	border-top-right-radius:0px;
	border-bottom-left-radius:0px;
	border: 2px solid;
	border-top: 0px solid;
	border-left: 0px solid;
}

QPushButton:hover{
	background-color:rgb(255, 197, 248);
}
QPushButton:pressed{
	border: 3px solid;
	border-bottom: 1px solid;
	border-right: 1px solid;
}
"""

SS_QUES_ON = """
QPushButton {
	border-color: rgba(0, 0 ,0,0.6);
	border-radius: 10px;
	background-color: rgb(255, 229, 255);
	text-align:left;
	border-bottom-right-radius:0px;
	border-bottom-left-radius:0px;
	border: 1px solid;
	border-right:3px solid;
	border-bottom: 0px solid;
}

QPushButton:hover{
	background-color:rgb(255, 197, 248);
}

QPushButton:pressed{
	border: 3px solid;
	border-bottom: 1px solid;
	border-right: 1px solid;
}
"""

class PoseType(Enum):
    THINKING = THINKING_POSES
    POINTING = POINTING_POSES


class DialogueType(Enum):
    THINKING = DIALOGUE_THINKING_LIST
    DOG = DIALOGUE_DOG_LIST
    CAT = DIALOGUE_CAT_LIST




last_category_dict = {
    DialogueType.THINKING: -1,
    DialogueType.CAT: -1,
    DialogueType.DOG: -1,
    PoseType.THINKING: -1,
    PoseType.POINTING: -1
}


def getPose(pose_name: str):
    return EVE_POSES_PATH + '\\' + pose_name


def getRandomPose(category: PoseType):
    global last_category_dict

    random_index = randint(0, len(category.value) - 1)
    while last_category_dict[category] == random_index:
        random_index = randint(0, len(category.value) - 1)

    last_category_dict[category] = random_index

    return getPose(category.value[random_index])


def getRandomDialogue(category: DialogueType):
    global last_category_dict

    random_index = randint(0, len(category.value) - 1)
    while last_category_dict[category] == random_index:
        random_index = randint(0, len(category.value) - 1)

    last_category_dict[category] = random_index

    return category.value[random_index]

