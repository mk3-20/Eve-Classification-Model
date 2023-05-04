import os
from pathlib import Path
from enum import Enum
from random import randint

CWD = os.getcwd()
PARENT_DIR_BACK_SLASH = Path(CWD).parent
PARENT_DIR = PARENT_DIR_BACK_SLASH.as_posix()
MAIN_WINDOW_TITLE = 'Eve - Classification Model'

ICON_QRC_PATH = ':/img/icon_img'

PREFERENCES_PATH = "save_file.txt"
EVE_POSES_PATH = "M:\\Mk_Coding\\lang_Python\\Projects\\Eve-Classification-Model\\Ui\\eve_poses"

EVE_CELEBRATING = "eve_celebrating.png"
EVE_EXPLAINING = "eve_explaining.png"
# EVE_HANDBAG = "eve_handbag.png"
EVE_HANDBOARD = "eve_handboard.png"
EVE_HANDFOLD = "eve_handfold.png"
EVE_POINTING = "eve_pointing.png"
EVE_THINKING = "eve_thinking.png"

POSES_LIST = [
    EVE_CELEBRATING,
    EVE_EXPLAINING,
    # EVE_HANDBAG,
    EVE_HANDBOARD,
    EVE_HANDFOLD,
    EVE_POINTING,
    EVE_THINKING
]
THINKING_POSES = [EVE_THINKING, EVE_HANDBOARD, EVE_HANDFOLD]
OTHER_POSES = [EVE_CELEBRATING, EVE_POINTING, EVE_EXPLAINING]

DIALOGUE_THINKING_LIST = ["I think that's...", "Ooo this one is surely..", "Hmmmm..", "Awww look at that..",
                          "Gotta think hard for this one.. "]
DIALOGUE_DOG_LIST = ["A cute lil doggo", "A dog", "A good doggy", "A DOG!!", "A Dawg", "doggy :))"]
DIALOGUE_CAT_LIST = ["A cat", "A Fluffy Kitty", "an El Gato", "A CAT!!", "kitty cat :D"]


class PoseType(Enum):
    THINKING = THINKING_POSES
    OTHER = OTHER_POSES


class DialogueType(Enum):
    THINKING = DIALOGUE_THINKING_LIST
    DOG = DIALOGUE_DOG_LIST
    CAT = DIALOGUE_CAT_LIST


last_category_dict = {
    DialogueType.THINKING: -1,
    DialogueType.CAT: -1,
    DialogueType.DOG: -1,
    PoseType.THINKING: -1,
    PoseType.OTHER: -1
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

# current_pose = 4
#
# def getNextPose():
#     global current_pose
#     current_pose = (current_pose + 1) % len(POSES_LIST)
#     return getPose(POSES_LIST[current_pose])
