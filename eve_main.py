"""
                                    ~~~ EVE ~~~
    A User-friendly graphical interface to demonstrate classification of dogs and cats
    using a Deep-Learning Model

    Aim:
    To build a GUI that:
    - Accepts input from user in the form of images (JPG/PNG/JPEG)
    - Classifies the images into two classes: Cats or Dogs
    - Asks the user to check the accuracy of the output

    Features:
    - Drag & drop support for inputting images
    - Ability to calculate the accuracy of the output
    - Answers to Frequently Asked Questions (FAQs)
    - Simple animations

    Tech Stack:
    - Python's PyQt6 framework for developing the GUI
    - Tensorflow Keras for developing the DL Model
    - Adobe for Ui designing and image assets

    Made by:
    Team ADAM
    - Ayushman Rawat (Ui/Ux Designer, Documentation)
    - Dhruv Neelesh Gupta (DL Model Developer)
    - Aman Shrivastava (DL Model Developer, Documentation)
    - Mukund Kukreja (GUI Developer, Tester)

"""
import json
# IMPORTS
import os
import shutil
import sys, time
import timeit
import traceback
from datetime import datetime
from enum import Enum
from random import sample

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from PyQt6.QtCore import QSize, Qt, QPoint, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, \
    QSequentialAnimationGroup, QTimer, QRect, QRunnable, pyqtSignal, QObject, pyqtSlot, QThreadPool
from PyQt6.QtGui import QPixmap, QMouseEvent, QFont, QIcon, QShortcut, QKeySequence, QMovie, QPainter
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QPushButton, QWidget, QInputDialog, QSizePolicy, QSpacerItem, QSplashScreen
from keras.preprocessing.image import ImageDataGenerator
from numpy import nan

import eve_strings as evestr


def convert_ui(ui_file_name: str, py_destination_path=None):  # ONLY FOR DEVELOPMENT (COMMENT OUT BEFORE MAKING EXE)
    """
    Converts given .ui in current working directory into file into .py file
    returns None
    """
    try:
        if py_destination_path is None:
            py_destination_path = ui_file_name
        os.system(f"python -m PyQt6.uic.pyuic -x {ui_file_name}.ui -o {py_destination_path}.py")
    except Exception as e:
        print(f"error function name = convert_ui , error = {e}")


main_ui_file_name = f'{evestr.CWD}/Ui/eve_ui'  # ONLY FOR DEVELOPMENT (COMMENT OUT BEFORE MAKING EXE)
convert_ui(main_ui_file_name, "eve_ui")  # ONLY FOR DEVELOPMENT (COMMENT OUT BEFORE MAKING EXE)

from eve_ui import Ui_MainWindow


class FrameType(Enum):
    NONE = 0
    ACCURACY = 1
    RESULTS = 2
    CREDITS = 3
    ASKEVE = 4


class Result:
    __input_dataframe: pd.DataFrame
    __classified_dataframe: pd.DataFrame
    __total_input: int = 0
    __cats_list: list[str] = []
    __dogs_list: list[str] = []
    __cats_count: int = 0
    __dogs_count: int = 0
    __correct_cats: int = 0
    __correct_dogs: int = 0
    __correct_total: int = 0
    __incorrect_cats: int = 0
    __incorrect_dogs: int = 0
    __incorrect_total: int = 0
    __output_dataframe_correct: list[str] = []
    __accuracy: float = 0.0

    def __init__(self, parent: QWidget, input_dataframe: pd.DataFrame):
        self.parent = parent
        self.__input_dataframe = input_dataframe
        self.__total_input = len(input_dataframe.index)

    def setClassifiedDataframe(self, classified_df: pd.DataFrame):
        self.__classified_dataframe = classified_df

    def setIncorrectList(self, incorrect_classifications: list[str]):
        self.__incorrect_total = len(incorrect_classifications)
        self.__correct_total = self.__total_input - self.__incorrect_total
        self.__incorrect_dogs = 0
        self.__accuracy = round((self.__correct_total / self.__total_input) * 100, 1)

        temp_counts = self.__classified_dataframe['Category'].value_counts()
        temp_cat_count = temp_counts[0] if 0 in temp_counts.index else 0
        temp_dog_count = self.__total_input - temp_cat_count

        for i in incorrect_classifications:
            original_category = \
                self.__classified_dataframe.loc[self.__classified_dataframe['Filename'] == i, 'Category'].iloc[0]
            if original_category:
                self.__incorrect_dogs += 1

            self.__classified_dataframe.loc[
                self.__classified_dataframe['Filename'] == i, 'Category'] = 0 if original_category else 1

        self.__incorrect_cats = self.__incorrect_total - self.__incorrect_dogs

        counts = self.__classified_dataframe['Category'].value_counts()
        self.__cats_count = counts[0] if 0 in counts.index else 0
        self.__dogs_count = self.__total_input - self.__cats_count

        self.__correct_cats = temp_cat_count - self.__incorrect_cats
        self.__correct_dogs = temp_dog_count - self.__incorrect_dogs

        if self.__cats_count:
            self.__cats_list = list(
                self.__classified_dataframe.loc[self.__classified_dataframe['Category'] == 0]['Filename'].values)
        if self.__dogs_count:
            self.__dogs_list = list(
                self.__classified_dataframe.loc[self.__classified_dataframe['Category'] == 1]['Filename'].values)

    def input_dataframe(self) -> pd.DataFrame:
        return self.__input_dataframe

    def classified_dataframe(self) -> pd.DataFrame:
        return self.__classified_dataframe

    def total_input(self) -> int:
        return self.__total_input

    def cats_list(self) -> list[str]:
        return self.__cats_list

    def dogs_list(self) -> list[str]:
        return self.__dogs_list

    def cats_count(self) -> int:
        return self.__cats_count

    def dogs_count(self) -> int:
        return self.__dogs_count

    def correct_cats(self) -> int:
        return self.__correct_cats

    def correct_dogs(self) -> int:
        return self.__correct_dogs

    def correct_total(self) -> int:
        return self.__correct_total

    def incorrect_cats(self) -> int:
        return self.__incorrect_cats

    def incorrect_dogs(self) -> int:
        return self.__incorrect_dogs

    def incorrect_total(self) -> int:
        return self.__incorrect_total

    def output_dataframe_correct(self) -> list[str]:
        return self.__output_dataframe_correct

    def accuracy(self) -> float:
        return self.__accuracy

    def saveResult(self):
        print("SAVING.....")
        save_folder_path = QFileDialog.getExistingDirectory(self.parent, "Select a save location",
                                                            preferences[evestr.KEY_SAVE_FOLDER])
        savePreferences(evestr.KEY_SAVE_FOLDER, save_folder_path)
        now = datetime.now().strftime("%d-%m-%Y %H_%M_%S")
        save_folder_now = save_folder_path + '/' + now
        cats_folder_path = save_folder_now + '/Cats'
        dogs_folder_path = save_folder_now + '/Dogs'
        if not os.path.exists(save_folder_now):
            os.makedirs(save_folder_now)
        if not os.path.exists(cats_folder_path):
            os.makedirs(cats_folder_path)
        if not os.path.exists(dogs_folder_path):
            os.makedirs(dogs_folder_path)
        with open(save_folder_now + '/Result_Summary.txt', 'w') as sf:
            sf.write(f"Total Images = {self.__total_input}\n")
            sf.write(f"Total Cats = {self.__cats_count}\n")
            sf.write(f"Total Dogs = {self.__dogs_count}\n\n")
            sf.write(f"Correctly Classified Cats = {self.__correct_cats}\n")
            sf.write(f"Correctly Classified Dogs = {self.__correct_dogs}\n")
            sf.write(f"Total Correct Classifications = {self.__correct_total}\n\n")
            sf.write(f"Incorrectly Classified Cats = {self.__incorrect_cats}\n")
            sf.write(f"Incorrectly Classified Dogs = {self.__incorrect_dogs}\n")
            sf.write(f"Total Incorrect Classifications = {self.__incorrect_total}\n\n")
            sf.write(f"Accuracy = {self.__accuracy}%\n")
        for c in self.__cats_list:
            shutil.copy2(c, cats_folder_path)
        for d in self.__dogs_list:
            shutil.copy2(d, dogs_folder_path)


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
        -> finished
            No data

        -> error
            tuple (exception_type, value, traceback.format_exc() )

        -> result
            object data returned from processing, anything

        -> progress
            int indicating % progress
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    """ Worker thread (Inherits from QRunnable) """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exception_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exception_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class MainScreen(QMainWindow, Ui_MainWindow):
    img_size = 225  # (for DL Model)
    batch_size = 32  # (for DL Model)
    testImageGenerator = ImageDataGenerator(rescale=1. / 255)  # for rescaling image (for DL Model)

    def __init__(self):  # Constructor
        super(MainScreen, self).__init__()  # Calling the super class's constructor
        self.setupUi(self)  # -> sets the imported ui
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)  # Making the window frameless i.e. removing the title bar
        self.setWindowTitle("Eve")  # Setting the app's title
        self.setAcceptDrops(True)  # Allowing the app to accept drag & drop events
        self.pushButton_importPics.setIcon(QIcon(evestr.IMAGE_ICON))
        self.pushButton_importMore.setIcon(QIcon(evestr.IMPORT_ICON))
        self.pushButton_resetImports.setIcon(QIcon(evestr.RESET_ICON))
        self.pushButton_collapseFrame.setIcon(QIcon(evestr.COLLAPSE_RIGHT_ICON))
        self.pushButton_expandAccuracy.setIcon(QIcon(evestr.COLLAPSE_LEFT_ICON))
        basket_pixmap_size = QSize(89, 89)
        self.label_catBasket.setPixmap(
            QPixmap(evestr.CAT_BASKET_IMG).scaled(basket_pixmap_size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))
        self.label_dogBasket.setPixmap(
            QPixmap(evestr.DOG_BASKET_IMG).scaled(basket_pixmap_size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation))

        # self.mainwindow_enlarge_animation = getPropertyAnimation(
        #     target=self,
        #     property_name=b'size',
        #     duration=750,
        #     start_value=QSize(0, 0),
        #     end_value=QSize(1060, 552),
        # )
        # self.mainwindow_enlarge_animation.start()

        self.verticalLayout_dog = QVBoxLayout()
        self.verticalLayout_dog.addLayout(QHBoxLayout())

        self.verticalLayout_cat = QVBoxLayout()
        self.verticalLayout_cat.addLayout(QHBoxLayout())

        self.offset = QPoint()  # To make the frameless window movable
        self.inputImagesDf: pd.DataFrame = pd.DataFrame(
            columns=['Filename'])  # Dataframe that'll include the filenames and the output class (0 for cat, 1 for dog)
        self.eve_model = tf.keras.models.load_model("eve_model.h5")
        self.incorrect_classifications_list: list[str] = []
        self.result_object: Result = Result(self, self.inputImagesDf)

        self.importPicsClicked = lambda _: self.setInputImages(
            # Called when user wants to select images from local storage
            QFileDialog.getOpenFileNames(parent=self, caption="Select Images",
                                         directory=preferences[evestr.KEY_IMAGE_FOLDER],
                                         filter="Images (*.jpg *.jpeg *.png)")[0]
        )

        def getSampleInput():
            if not self.pushButton_importPics.isVisible() or not self.pushButton_importPics.isEnabled():
                return
            folder_path, ok = QInputDialog.getText(self, 'Folder Path', 'Enter Images Folder Path: ',
                                                   text=prev_folder_path)
            if ok:
                if os.path.exists(folder_path):
                    setSampleImageList(folder_path)
                else:
                    QMessageBox.critical(self, "Error", "Invalid Path *_*")
                    return

                num, ok = QInputDialog.getText(self, 'Random Sample Input', 'Enter Sample Size: ')
                if ok:
                    try:
                        print(f"{sample_image_files = }, {num = }")
                        if sample_image_files:
                            num = int(num) % (len(sample_image_files) + 1)
                            self.setInputImages(sample(sample_image_files, num))
                        else:
                            QMessageBox.information(self, "No Sample", "No images selected..")

                    except Exception as e:
                        QMessageBox.critical(self, "Invalid value *_*", str(e))

        self.ctrl_i_pressed = QShortcut(QKeySequence("Ctrl+I"), self)
        self.ctrl_i_pressed.activated.connect(getSampleInput)

        self.enter_pressed = QShortcut(QKeySequence("Return"), self)
        self.enter_pressed.activated.connect(self.enterClicked)

        # Connecting Signals (button clicks) and Slots (functions)
        self.pushButton_importPics.clicked.connect(self.importPicsClicked)
        self.pushButton_importMore.clicked.connect(self.importPicsClicked)
        self.pushButton_resetImports.clicked.connect(self.resetImportsClicked)
        self.pushButton_classify.clicked.connect(self.classifyClicked)
        self.pushButton_close.clicked.connect(self.deleteLater)
        self.pushButton_minimize.clicked.connect(self.showMinimized)
        self.pushButton_collapseFrame.clicked.connect(self.collapseFrameClicked)
        self.pushButton_expandAccuracy.clicked.connect(self.expandAccuracyClicked)
        self.pushButton_speedUp.clicked.connect(self.speedBtnClicked)
        self.pushButton_slowDown.clicked.connect(self.speedBtnClicked)
        self.pushButton_accuracy.clicked.connect(self.accuracyBtnClicked5)
        self.pushButton_save_output.clicked.connect(self.saveResultClicked)
        self.pushButton_thankyou.clicked.connect(self.thankYouEveClicked)
        self.pushButton_credits.clicked.connect(self.creditsClicked)
        self.pushButton_askEve.clicked.connect(self.askEveClicked)
        self.pushButton_quesObjective.clicked.connect(self.questionClicked)

        # ANIMATIONS
        bubble_anim_factor = 10
        bubble_anim_timing = 100

        self.label_bubble_enlarge_animation = getPropertyAnimation(
            target=self.label_speech_bubble,
            property_name=b'size',
            duration=bubble_anim_timing,
            start_value=self.label_speech_bubble.size(),
            end_value=self.label_speech_bubble.size() + QSize(bubble_anim_factor, bubble_anim_factor)
        )

        self.label_bubble_shrink_animation = getPropertyAnimation(
            target=self.label_speech_bubble,
            property_name=b'size',
            duration=bubble_anim_timing,
            start_value=self.label_speech_bubble.size() + QSize(bubble_anim_factor, bubble_anim_factor),
            end_value=self.label_speech_bubble.size()
        )

        self.label_dialogue_move_animation = getPropertyAnimation(
            target=self.label_dialogue,
            property_name=b'pos',
            duration=bubble_anim_timing,
            start_value=self.label_dialogue.pos(),
            end_value=self.label_dialogue.pos() + QPoint(bubble_anim_factor, bubble_anim_factor)
        )

        self.label_dialogue_move_back_animation = getPropertyAnimation(
            target=self.label_dialogue,
            property_name=b'pos',
            duration=bubble_anim_timing,
            start_value=self.label_dialogue.pos() + QPoint(bubble_anim_factor, bubble_anim_factor),
            end_value=self.label_dialogue.pos()
        )

        self.label_bubble_dialogue_enlarge_grp = QParallelAnimationGroup()
        self.label_bubble_dialogue_enlarge_grp.addAnimation(self.label_bubble_enlarge_animation)
        self.label_bubble_dialogue_enlarge_grp.addAnimation(self.label_dialogue_move_animation)

        self.label_bubble_dialogue_shrink_grp = QParallelAnimationGroup()
        self.label_bubble_dialogue_shrink_grp.addAnimation(self.label_bubble_shrink_animation)
        self.label_bubble_dialogue_shrink_grp.addAnimation(self.label_dialogue_move_back_animation)

        self.label_bubble_pop_seq = QSequentialAnimationGroup()
        self.label_bubble_pop_seq.addAnimation(self.label_bubble_dialogue_enlarge_grp)
        self.label_bubble_pop_seq.addAnimation(self.label_bubble_dialogue_shrink_grp)

        basket_anim_timing = 100
        btn_anim_factor = 10

        self.basket_catdog_enlarge_animation = getPropertyAnimation(
            target=None,
            property_name=b'size',
            duration=basket_anim_timing,
            start_value=self.label_catBasket.size(),
            end_value=self.label_catBasket.size() + QSize(btn_anim_factor, btn_anim_factor)
        )

        self.basket_catdog_shrink_animation = getPropertyAnimation(
            target=None,
            property_name=b'size',
            duration=basket_anim_timing,
            start_value=self.label_catBasket.size() + QSize(btn_anim_factor, btn_anim_factor),
            end_value=self.label_catBasket.size()
        )

        self.btn_catdog_pop_seq = QSequentialAnimationGroup()
        self.btn_catdog_pop_seq.addAnimation(self.basket_catdog_enlarge_animation)
        self.btn_catdog_pop_seq.addAnimation(self.basket_catdog_shrink_animation)

        input_img_anim_vertical_timing = evestr.DEFAULT_ANIM_INPUT_IMAGE_VERTICAL
        input_img_anim_horizontal_timing = evestr.DEFAULT_ANIM_INPUT_IMAGE_HORIZONTAL

        self.input_anim_move_vertical = QPropertyAnimation()
        self.input_anim_move_vertical.setPropertyName(b'pos')
        self.input_anim_move_vertical.setDuration(input_img_anim_vertical_timing)
        self.input_anim_move_vertical.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.input_anim_move_horizontal = QPropertyAnimation()
        self.input_anim_move_horizontal.setPropertyName(b'pos')
        self.input_anim_move_horizontal.setDuration(input_img_anim_horizontal_timing)
        self.input_anim_move_horizontal.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.input_anim_shrink = QPropertyAnimation()
        self.input_anim_shrink.setPropertyName(b'size')
        self.input_anim_shrink.setEndValue(QSize(0, 0))
        self.input_anim_shrink.setDuration(input_img_anim_horizontal_timing)

        self.input_anim_grp = QParallelAnimationGroup()
        self.input_anim_grp.addAnimation(self.input_anim_move_horizontal)
        self.input_anim_grp.addAnimation(self.input_anim_shrink)

        self.input_anim_seq = QSequentialAnimationGroup()
        self.input_anim_seq.addAnimation(self.input_anim_move_vertical)
        self.input_anim_seq.addAnimation(self.input_anim_grp)

        self.input_anim_seq.finished.connect(self.endClassificationAnimation3)

        frame_expand_timing = evestr.DEFAULT_FRAME_EXPAND_TIMING
        frame_collapse_timing = evestr.DEFAULT_FRAME_COLLAPSE_TIMING
        eve_move_anim_factor = QPoint(210, 0)

        def getExpandFramePropAnim(t, factor=eve_move_anim_factor, timing=frame_expand_timing):
            return getPropertyAnimation(target=t, property_name=b'pos',
                                        duration=timing, start_value=t.pos(),
                                        end_value=t.pos() - factor,
                                        easing_curve=QEasingCurve.Type.OutCurve)

        def getCollapseFramePropAnim(t, factor=eve_move_anim_factor, timing=frame_collapse_timing):
            return getPropertyAnimation(target=t, property_name=b'pos',
                                        duration=timing, start_value=t.pos() - factor,
                                        end_value=t.pos(),
                                        easing_curve=QEasingCurve.Type.OutCurve)

        def getFrameAnimGrp(frame, collapse: bool = False) -> QParallelAnimationGroup:
            fn = getExpandFramePropAnim
            if collapse:
                fn = getCollapseFramePropAnim

            move_eve = fn(self.label_eve)
            move_bubble = fn(self.label_speech_bubble)
            move_dialogue = fn(self.label_dialogue)
            if frame == self.frame_results and collapse:
                move_frame = fn(frame, QPoint(frame.width(), 0), evestr.DEFAULT_RESULTS_COLLAPSE_TIMING)
            else:
                move_frame = fn(frame, QPoint(frame.width(), 0))

            frame_anim_grp = QParallelAnimationGroup()
            frame_anim_grp.addAnimation(move_frame)
            frame_anim_grp.addAnimation(move_eve)
            frame_anim_grp.addAnimation(move_bubble)
            frame_anim_grp.addAnimation(move_dialogue)
            if not collapse:
                frame_anim_grp.finished.connect(self.pushButton_collapseFrame.show)

            return frame_anim_grp

        self.expand_accuracy_anim_grp = getFrameAnimGrp(self.frame_accuracy)
        self.collapse_accuracy_anim_grp = getFrameAnimGrp(self.frame_accuracy, collapse=True)
        self.collapse_accuracy_anim_grp.finished.connect(self.pushButton_expandAccuracy.show)

        self.expand_results_anim_grp = QParallelAnimationGroup()
        self.expand_results_anim_grp.addAnimation(
            getExpandFramePropAnim(self.frame_results, QPoint(self.frame_results.width(), 0),
                                   evestr.DEFAULT_RESULTS_EXPAND_TIMING))
        self.expand_results_anim_grp.addAnimation(
            getCollapseFramePropAnim(self.frame_accuracy, QPoint(self.frame_accuracy.width(), 0)))
        self.collapse_results_anim_grp = getFrameAnimGrp(self.frame_results, True)

        self.expand_credits_anim_grp = getFrameAnimGrp(self.frame_credits)
        self.collapse_credits_anim_grp = getFrameAnimGrp(self.frame_credits, True)

        self.expand_askEve_anim_grp = getFrameAnimGrp(self.frame_askEve)
        self.collapse_askEve_anim_grp = getFrameAnimGrp(self.frame_askEve, True)

        # Making the classify button invisible at first
        self.pushButton_classify.hide()
        self.pushButton_importMore.hide()
        self.pushButton_resetImports.hide()
        self.pushButton_collapseFrame.hide()
        self.pushButton_expandAccuracy.hide()
        self.label_catBasket.hide()
        self.label_dogBasket.hide()
        self.pushButton_speedUp.hide()
        self.pushButton_slowDown.hide()
        self.pushButton_askEve.show()
        self.pushButton_credits.show()

        self.current_frame: FrameType = FrameType.NONE
        self.just_identified_dog = False
        self.first_image_flag = True
        self.same_batch = False
        self.anim_speed_multiplier = 1
        QTimer.singleShot(evestr.DEFAULT_DELAY_INTRO, self.intro)
        self.model_worker = None
        self.threadpool = QThreadPool()
        self.animation_running = False
        self.current_img_index = 0
        self.last_ques_opened = None
        self.label_ansObjective.hide()
        self.qa_btn_label_dict: dict[QPushButton, tuple[QLabel, str]] = {
            self.pushButton_quesObjective: (self.label_ansObjective, "My Objective is to...")}
        self.setAskEveQA()
        self.pushButton_askEve.hide()
        self.pushButton_credits.hide()
        self.pushButton_importPics.hide()

    def intro(self):
        self.setDialogue("I'm Eve :D")
        self.pushButton_askEve.show()
        self.pushButton_credits.show()
        self.pushButton_importPics.show()

    def creditsClicked(self):
        self.setPose(evestr.EVE_POINTING)
        self.setDialogue("This awesome team was behind my development! :)")
        self.expand_credits_anim_grp.start()
        self.current_frame = FrameType.CREDITS

    def askEveClicked(self):
        self.setPose(evestr.EVE_HANDBOARD)
        self.setDialogue("Here are some frequently asked questions...")
        self.expand_askEve_anim_grp.start()
        self.current_frame = FrameType.ASKEVE

    def setAskEveQA(self):
        for ques_key in ask_eve_qa_dict:
            ques = ask_eve_qa_dict[ques_key][0]
            ans_intro = ask_eve_qa_dict[ques_key][1]
            ans = ask_eve_qa_dict[ques_key][2]
            horizontalLayout = QHBoxLayout()
            horizontalLayout.setObjectName(f"horizontalLayout_{ques_key}")
            spacerItemLeft = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            horizontalLayout.addItem(spacerItemLeft)

            verticalLayout = QVBoxLayout()
            verticalLayout.setSpacing(0)
            verticalLayout.setObjectName(f"verticalLayout_{ques_key}")
            pushButton_ques = QPushButton(parent=self.scrollAreaWidgetContents_askEve)
            sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)

            sizePolicy.setHeightForWidth(self.pushButton_quesObjective.sizePolicy().hasHeightForWidth())
            pushButton_ques.setSizePolicy(sizePolicy)
            pushButton_ques.setMinimumSize(QSize(609, 41))
            pushButton_ques.setMaximumSize(QSize(609, 41))
            font = QFont()
            font.setFamily("Segoe UI Variable Small Semibol")
            font.setPointSize(11)
            font.setBold(False)
            font.setWeight(50)
            pushButton_ques.setFont(font)
            pushButton_ques.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            pushButton_ques.setStyleSheet(evestr.SS_QUES_OFF)
            drop_down_icon = QIcon()
            drop_down_icon.addPixmap(QPixmap(evestr.DROP_DOWN_ICON), QIcon.Mode.Normal, QIcon.State.Off)
            pushButton_ques.setIcon(drop_down_icon)
            pushButton_ques.setIconSize(QSize(30, 30))
            pushButton_ques.setObjectName(f"pushButton_{ques_key}")
            pushButton_ques.setText(f"   Q. {ques}")
            verticalLayout.addWidget(pushButton_ques)

            label_ans = QLabel(parent=self.scrollAreaWidgetContents_askEve)
            sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.label_ansObjective.sizePolicy().hasHeightForWidth())
            label_ans.setSizePolicy(sizePolicy)
            label_ans.setMinimumSize(QSize(609, 0))
            label_ans.setMaximumSize(QSize(609, 16777215))
            font = QFont()
            font.setFamily("8514oem")
            font.setPointSize(10)
            label_ans.setFont(font)
            label_ans.setStyleSheet(
                "border: 3px solid;\nborder-top: 0px solid;\nborder-left: 1px solid;\npadding:10px;")
            label_ans.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label_ans.setWordWrap(True)
            label_ans.setObjectName(f"label_ans{ques_key[4:]}")
            label_ans.setText(ans)
            verticalLayout.addWidget(label_ans)
            horizontalLayout.addLayout(verticalLayout)

            spacerItemRight = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            horizontalLayout.addItem(spacerItemRight)
            horizontalLayout.setStretch(0, 1)
            horizontalLayout.setStretch(2, 1)
            self.verticalLayout_13.addLayout(horizontalLayout)
            label_ans.hide()
            pushButton_ques.clicked.connect(self.questionClicked)
            self.qa_btn_label_dict[pushButton_ques] = (label_ans, ans_intro)
        spacerItemBottom = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.verticalLayout_13.addItem(spacerItemBottom)
        self.verticalLayout_13.setStretch(-1, 2)

    def questionClicked(self):

        ques_btn: QPushButton = self.sender()
        ans_label: QLabel = self.qa_btn_label_dict[ques_btn][0]
        ans_intro: str = self.qa_btn_label_dict[ques_btn][1]
        if ans_label.isVisible():  # ANSWER IS ALREADY OPEN
            self.setPose(evestr.EVE_THINKING)
            self.setDialogue("Wanna ask more?")
            ques_btn.setStyleSheet(evestr.SS_QUES_OFF)
            ques_btn.setIcon(QIcon(evestr.DROP_DOWN_ICON))
            ans_label.hide()
        else:  # HAVE TO OPEN THE ANSWER
            if self.last_ques_opened is not None and self.last_ques_opened != ques_btn and \
                    self.qa_btn_label_dict[self.last_ques_opened][0].isVisible():
                self.last_ques_opened.click()
            self.setPose(evestr.EVE_EXPLAINING)
            self.setDialogue(ans_intro)
            ques_btn.setStyleSheet(evestr.SS_QUES_ON)
            ques_btn.setIconSize(QSize(30, 30))
            ques_btn.setIcon(QIcon(evestr.DROP_UP_ICON))
            self.last_ques_opened = ques_btn

            # ans_label.setFixedHeight(evestr.ANSWER_HEIGHT)
            # ans_label.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Expanding)
            ans_label.adjustSize()
            ans_label.show()

    def showSpeechBubble(self, show: bool):
        self.label_dialogue.setVisible(show)
        self.label_speech_bubble.setVisible(show)

    def setDialogue(self, dialogue: str, font_size: int = 18, font: str = "Gill Sans MT Condensed", ):
        self.label_dialogue.setFont(QFont(font, font_size))

        og_pos = self.label_dialogue.pos()
        extended_pos = self.label_dialogue.pos() + QPoint(10, 10)
        self.label_dialogue_move_animation.setStartValue(og_pos)
        self.label_dialogue_move_animation.setEndValue(extended_pos)
        self.label_dialogue_move_back_animation.setStartValue(extended_pos)
        self.label_dialogue_move_back_animation.setEndValue(og_pos)
        self.label_bubble_pop_seq.start()
        self.label_dialogue.setText(dialogue)

    def setPose(self, pose: str):
        self.label_eve.setPixmap(QPixmap(evestr.getPose(pose)))

    def setInputImages(self, path_list: list[str]):  # TO UPDATE THE INPUT DATAFRAME (adding more input files)
        if path_list:
            self.pushButton_importPics.hide()
            self.pushButton_importMore.show()
            self.pushButton_resetImports.show()
            self.pushButton_classify.show()
            self.pushButton_expandAccuracy.hide()
            self.pushButton_askEve.hide()
            self.pushButton_credits.hide()
            self.showSpeechBubble(False)

            savePreferences(evestr.KEY_IMAGE_FOLDER, os.path.dirname(path_list[-1]) + '/')
            self.inputImagesDf = pd.concat([self.inputImagesDf, pd.DataFrame({'Filename': path_list})]).reset_index(
                drop=True)
            self.result_object = Result(self, self.inputImagesDf)

            print("INPUT DF: ", self.inputImagesDf,
                  f"\n{len(self.inputImagesDf.index) = }, {self.current_img_index = }")
            if len(self.inputImagesDf.index) > self.current_img_index:
                img = self.inputImagesDf.iloc[self.current_img_index]['Filename']
                print(f"first img = {img}")
                resize = self.label_inputImages.height() - 50
                img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
                self.label_inputImages.setPixmap(img_pixmap)
                self.label_inputImages.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def enterClicked(self):
        if self.current_frame == FrameType.NONE:
            if self.pushButton_importPics.isVisible():
                self.pushButton_importPics.click()
            elif self.pushButton_classify.isVisible():
                self.pushButton_classify.click()

        elif self.current_frame == FrameType.CREDITS:
            self.pushButton_collapseFrame.click()
        elif self.current_frame == FrameType.ACCURACY:
            self.pushButton_accuracy.click()
        elif self.current_frame == FrameType.RESULTS:
            self.pushButton_thankyou.click()

    def speedBtnClicked(self):
        speed_up_btn = self.sender() == self.pushButton_speedUp
        if speed_up_btn:
            self.anim_speed_multiplier *= 0.5
        else:
            self.anim_speed_multiplier = 1
        # self.anim_speed_multiplier *= (0.5 if speed_up_btn else 2)

        self.input_anim_move_horizontal.setDuration(
            int(evestr.DEFAULT_ANIM_INPUT_IMAGE_HORIZONTAL * self.anim_speed_multiplier))
        self.input_anim_move_vertical.setDuration(
            int(evestr.DEFAULT_ANIM_INPUT_IMAGE_VERTICAL * self.anim_speed_multiplier))
        self.input_anim_shrink.setDuration(int(evestr.DEFAULT_ANIM_INPUT_IMAGE_HORIZONTAL * self.anim_speed_multiplier))

    def resetImportsClicked(self):
        self.pushButton_importPics.show()
        self.pushButton_importMore.hide()
        self.pushButton_resetImports.hide()
        self.pushButton_classify.hide()
        self.showSpeechBubble(True)
        self.pushButton_askEve.show()
        self.pushButton_credits.show()
        self.inputImagesDf = pd.DataFrame(columns=['Filename'])
        if self.same_batch:
            self.pushButton_expandAccuracy.show()
            self.setDialogue("You can import more images or check my accuracy")
        else:
            self.setDialogue("Please give me some images to work with :D")

    def runModel(self, progress_callback):
        try:
            # BATCH-WISE
            total_input_images = len(self.inputImagesDf.index)
            batch_one_size = 0
            first_nan_index = 0
            if 'Category' in self.inputImagesDf.columns:
                first_nan_index = self.inputImagesDf['Category'].isnull().idxmax()
                total_input_images -= first_nan_index

            if total_input_images > 4:
                batch_one_size = (((total_input_images // 100) + 1) * 4 if total_input_images > 200 else 4)
                print("batch_one_size = ", batch_one_size)
                batch_one_size += first_nan_index

            threshold = 0.5  # >0.5 = 1 (Dog)  | <0.5 = 0 (Cat)

            start_time = timeit.default_timer()
            if batch_one_size > 0:
                batch1 = self.testImageGenerator.flow_from_dataframe(
                    self.inputImagesDf.loc[first_nan_index:batch_one_size - 1, :], "",
                    x_col='Filename', y_col=None,
                    class_mode=None, batch_size=self.batch_size,
                    target_size=(self.img_size, self.img_size),
                    shuffle=False)

                predictions1 = self.eve_model.predict(batch1,
                                                      steps=np.ceil(
                                                          self.inputImagesDf.loc[first_nan_index:batch_one_size - 1,
                                                          :].shape[
                                                              0] / self.batch_size))  # numpy.ndarray [[val] [val] ..]

                self.inputImagesDf.loc[first_nan_index:batch_one_size - 1, 'Category'] = np.where(
                    predictions1 > threshold, 1, 0)
                progress_callback.emit(True)

            batch2 = self.testImageGenerator.flow_from_dataframe(self.inputImagesDf.loc[batch_one_size:, :], "",
                                                                 x_col='Filename', y_col=None,
                                                                 class_mode=None, batch_size=self.batch_size,
                                                                 target_size=(self.img_size, self.img_size),
                                                                 shuffle=False)

            predictions2 = self.eve_model.predict(batch2,
                                                  steps=np.ceil(
                                                      self.inputImagesDf.loc[batch_one_size:, :].shape[
                                                          0] / self.batch_size))

            self.inputImagesDf.loc[batch_one_size:, 'Category'] = np.where(predictions2 > threshold, 1, 0)
            end_time = timeit.default_timer()
            progress_callback.emit(True)
            print(f"BATCH-WISE TIME for {total_input_images} images = {end_time - start_time} seconds")

        except Exception as e:
            QMessageBox.information(self, "ERROR while classification :((", str(e))

    def classifyClicked(self):
        """
            ACTUAL Classification of the input images using DL Model and setting the output in the DATAFRAME
        """

        self.pushButton_classify.hide()
        self.pushButton_importMore.hide()
        self.pushButton_resetImports.hide()
        self.label_catBasket.show()
        self.label_dogBasket.show()
        self.showSpeechBubble(True)
        self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.THINKING)))
        self.setDialogue("Hmm.. Let's see what we have")
        if not self.same_batch:
            self.current_img_index = 0

        def progress_batch_one_done(batch_one_done: bool):
            print("BATCH ONE DONE? ", batch_one_done)
            if batch_one_done and not self.animation_running: self.beginAnimation0()

        def enableSpeedControl():
            self.pushButton_speedUp.setDisabled(False)
            self.pushButton_slowDown.setDisabled(False)
            self.result_object.setClassifiedDataframe(self.inputImagesDf)
            print(f"TOOK {self.current_img_index} animations for complete classifications")

        self.pushButton_speedUp.setDisabled(True)
        self.pushButton_slowDown.setDisabled(True)

        self.model_worker = Worker(self.runModel)
        self.model_worker.signals.finished.connect(enableSpeedControl)
        self.model_worker.signals.progress.connect(progress_batch_one_done)
        self.threadpool.start(self.model_worker)

    def beginAnimation0(self):
        self.animation_running = True

        self.pushButton_speedUp.show()
        self.pushButton_slowDown.show()
        self.thinkingPauseAnimation1()

    def thinkingPauseAnimation1(self):
        self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.THINKING)))
        self.label_dialogue.setText(evestr.getRandomDialogue(evestr.DialogueType.THINKING))
        QTimer.singleShot(int(evestr.DEFAULT_DELAY_CLASSIFICATION * self.anim_speed_multiplier),
                          self.classificationAnimation2)

    def classificationAnimation2(self):
        if self.input_anim_move_horizontal.targetObject() is not None:
            self.input_anim_move_horizontal.targetObject().deleteLater()

        if len(self.inputImagesDf.index) < self.current_img_index:
            return

        img_path, img_is_dog = self.inputImagesDf.iloc[self.current_img_index]
        if (img_is_dog != 0 and img_is_dog != 1) or img_is_dog == nan:
            print("PAUSING.....")
            self.pushButton_slowDown.click()
            self.thinkingPauseAnimation1()
            return

        self.same_batch = True

        self.just_identified_dog = img_is_dog
        self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.POINTING)))
        self.first_image_flag = False
        self.setDialogue(
            evestr.getRandomDialogue(evestr.DialogueType.DOG if img_is_dog else evestr.DialogueType.CAT), font_size=25)
        self.current_img_index += 1

        # self.inputImagesDf = self.inputImagesDf.iloc[1:, :]  # popping first row from the dataframe

        def updateIncorrectList(label: QLabel):
            incorrect_img_path = label.objectName()
            selected = False
            if incorrect_img_path in self.incorrect_classifications_list:
                self.incorrect_classifications_list.remove(incorrect_img_path)
            else:
                selected = True
                self.incorrect_classifications_list.append(incorrect_img_path)
            label.setProperty('selected', selected)
            label.style().unpolish(label)
            label.style().polish(label)
            if self.incorrect_classifications_list:
                self.pushButton_accuracy.setText(
                    f" You got {len(self.incorrect_classifications_list)} of them incorrect, Eve.. ")
            else:
                self.pushButton_accuracy.setText(" All classifications are correct, Eve! :D ")

        output_image_size = QSize(140, 120)
        img_pixmap = QPixmap(img_path).scaled(output_image_size, Qt.AspectRatioMode.IgnoreAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)

        img_label = QLabel()
        img_label.setFixedSize(output_image_size + QSize(15, 15))
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setPixmap(img_pixmap)
        img_label.setObjectName(img_path)
        img_label.setStyleSheet(
            'QLabel{background-color:white; border:3px solid white;} QLabel[selected = "false"]{border-color: white;}QLabel[selected = "true"]{border-color: rgb(255, 0, 127);} QLabel:hover{border:2px solid; border-color: rgb(255, 0, 127);} QLabel:pressed{border:3px solid;}')
        img_label.mouseDoubleClickEvent = lambda _: Image.open(img_path).show(title=img_path)
        img_label.mousePressEvent = lambda _: updateIncorrectList(img_label)

        img_label_animated = QLabel(parent=self.centralwidget)
        img_label_animated.resize(self.label_inputImages.size())
        img_label_animated.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label_animated.setPixmap(img_pixmap)
        img_label_animated.setScaledContents(True)

        x_offset = (self.label_inputImages.pos().x()) + (self.label_inputImages.size().width() // 2) - (
            0 if img_is_dog else img_label_animated.size().width())

        y_offset = (self.label_inputImages.pos().y()) - int(self.label_inputImages.size().height() / 1.5)

        def get_basket_center(btn: QPushButton):
            # return QPoint(btn.pos().x() + (btn.width() // 2), btn.pos().y() + (btn.height() // 2))
            return QPoint(btn.pos().x() + (btn.width() // 2), btn.pos().y())

        img_label_animated.move(self.label_inputImages.pos())
        img_label_animated.show()

        self.input_anim_move_vertical.setStartValue(img_label_animated.pos())
        self.input_anim_move_vertical.setEndValue(QPoint(x_offset, y_offset))
        self.input_anim_move_vertical.setTargetObject(img_label_animated)

        self.input_anim_move_horizontal.setTargetObject(img_label_animated)
        self.input_anim_shrink.setTargetObject(img_label_animated)
        self.input_anim_shrink.setStartValue(img_label_animated.size())
        self.input_anim_move_horizontal.setStartValue(QPoint(x_offset, y_offset))
        self.input_anim_move_horizontal.setEndValue(
            get_basket_center(self.label_dogBasket if img_is_dog else self.label_catBasket))
        self.input_anim_seq.start()

        insertInVLayout(self.verticalLayout_dog if img_is_dog else self.verticalLayout_cat, img_label)
        if len(self.inputImagesDf.index) > self.current_img_index:
            img = self.inputImagesDf.iloc[self.current_img_index]['Filename']
            resize = self.label_inputImages.height() - 50
            img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label_inputImages.setPixmap(img_pixmap)
        else:
            self.label_inputImages.setPixmap(QPixmap())

    def endClassificationAnimation3(self):
        if not self.first_image_flag:
            target_btn = self.label_dogBasket if self.just_identified_dog else self.label_catBasket
            self.basket_catdog_enlarge_animation.setTargetObject(target_btn)
            self.basket_catdog_shrink_animation.setTargetObject(target_btn)
            self.btn_catdog_pop_seq.start()

        if len(self.inputImagesDf.index) > self.current_img_index:
            QTimer.singleShot(int(evestr.DEFAULT_DELAY_THINKING * self.anim_speed_multiplier),
                              self.thinkingPauseAnimation1)

        else:
            self.setPose(evestr.EVE_CELEBRATING)

            self.setDialogue("That was fun ^_^")
            self.first_image_flag = True
            self.animation_running = False

            QTimer.singleShot(int(evestr.DEFAULT_DELAY_ACCURACY * self.anim_speed_multiplier), self.accuracyAnimation4)

    def accuracyAnimation4(self):
        self.label_catBasket.hide()
        self.label_dogBasket.hide()
        self.pushButton_speedUp.hide()
        self.pushButton_slowDown.hide()
        self.pushButton_importPics.setDisabled(True)
        self.setPose(evestr.EVE_HANDBOARD)
        self.setDialogue("Now let's see how accurate I was..")
        self.expand_accuracy_anim_grp.start()
        self.current_frame = FrameType.ACCURACY
        self.scrollAreaWidgetContents_cats.setLayout(self.verticalLayout_cat)
        self.scrollAreaWidgetContents_dogs.setLayout(self.verticalLayout_dog)

    def accuracyBtnClicked5(self):
        self.same_batch = False
        self.result_object.setIncorrectList(self.incorrect_classifications_list)

        self.pushButton_collapseFrame.hide()
        self.expand_results_anim_grp.start()
        self.current_frame = FrameType.RESULTS

        self.label_value_total_images.setText(str(self.result_object.total_input()))
        self.label_value_correct_cats.setText(str(self.result_object.correct_cats()))
        self.label_value_correct_dogs.setText(str(self.result_object.correct_dogs()))
        self.label_value_correct_total.setText(str(self.result_object.correct_total()))
        self.label_value_incorrect_cats.setText(str(self.result_object.incorrect_cats()))
        self.label_value_incorrect_dogs.setText(str(self.result_object.incorrect_dogs()))
        self.label_value_incorrect_total.setText(str(self.result_object.incorrect_total()))
        self.label_value_accuracy.setText(str(self.result_object.accuracy()) + "%")

        self.setPose(evestr.EVE_HANDFOLD if self.result_object.accuracy() < 100 else evestr.EVE_CELEBRATING)

        self.setDialogue("I will try to do better next time.." if self.result_object.accuracy() < 100 else "Yay!!")

    def saveResultClicked(self):
        self.result_object.saveResult()
        self.setDialogue("Result Saved!", 25)

    def thankYouEveClicked(self):

        self.collapse_results_anim_grp.start()
        self.current_frame = FrameType.NONE

        def lastDialogue():
            self.setPose(evestr.EVE_THINKING)
            self.setDialogue("You can select images or ask me questions ^-^")

        self.collapse_results_anim_grp.finished.connect(lastDialogue)
        self.reset()

    def collapseFrameClicked(self):
        self.pushButton_importPics.show()
        self.pushButton_collapseFrame.hide()
        self.pushButton_importPics.setDisabled(False)
        self.setPose(evestr.EVE_HANDFOLD)

        if self.current_frame == FrameType.ACCURACY:
            self.setDialogue("You can import more images or check my accuracy")
            self.collapse_accuracy_anim_grp.start()
            self.pushButton_askEve.show()
            self.pushButton_credits.show()

        elif self.current_frame == FrameType.CREDITS:
            self.setDialogue("You can import images or ask me questions :D")
            self.collapse_credits_anim_grp.start()

        elif self.current_frame == FrameType.ASKEVE:
            self.setDialogue("You can try to import some pics and watch me work :))")
            self.collapse_askEve_anim_grp.start()

        self.current_frame = FrameType.NONE

    def expandAccuracyClicked(self):
        self.pushButton_askEve.hide()
        self.pushButton_credits.hide()
        self.pushButton_expandAccuracy.hide()
        self.setPose(evestr.EVE_HANDBOARD)
        self.setDialogue("Please select the pictures which I classified incorrectly")
        self.expand_accuracy_anim_grp.start()
        self.current_frame = FrameType.ACCURACY

    def reset(self):
        self.current_img_index = 0
        self.pushButton_importPics.setDisabled(False)
        self.pushButton_importPics.show()
        self.pushButton_askEve.show()
        self.pushButton_credits.show()
        self.inputImagesDf = pd.DataFrame(columns=['Filename'])
        self.incorrect_classifications_list.clear()
        self.pushButton_slowDown.click()

        self.verticalLayout_cat.deleteLater()
        self.verticalLayout_dog.deleteLater()

        self.verticalLayout_dog = QVBoxLayout()
        self.verticalLayout_dog.addLayout(QHBoxLayout())

        self.verticalLayout_cat = QVBoxLayout()
        self.verticalLayout_cat.addLayout(QHBoxLayout())

        self.scrollAreaWidgetContents_cats.deleteLater()
        self.scrollAreaWidgetContents_cats = QWidget()
        self.scrollAreaWidgetContents_cats.setGeometry(QRect(0, 0, 667, 185))
        self.scrollAreaWidgetContents_cats.setStyleSheet("QWidget#scrollAreaWidgetContents_cats{\n"
                                                         "background-color: rgba(255, 255, 255,0.7);\n"
                                                         "}")
        self.scrollAreaWidgetContents_cats.setObjectName("scrollAreaWidgetContents_cats")
        self.scrollArea_cats.setWidget(self.scrollAreaWidgetContents_cats)

        self.scrollAreaWidgetContents_dogs.deleteLater()
        self.scrollAreaWidgetContents_dogs = QWidget()
        self.scrollAreaWidgetContents_dogs.setGeometry(QRect(0, 0, 667, 184))
        self.scrollAreaWidgetContents_dogs.setStyleSheet("QWidget#scrollAreaWidgetContents_dogs{\n"
                                                         "border-top-left-radius: 0px;\n"
                                                         "border-top-right-radius: 0px;\n"
                                                         "background-color: rgba(255, 255, 255,0.7);\n"
                                                         "}")
        self.scrollAreaWidgetContents_dogs.setObjectName("scrollAreaWidgetContents_dogs")
        self.scrollArea_dogs.setWidget(self.scrollAreaWidgetContents_dogs)

    # OVERRIDDEN FUNCTIONS
    def mousePressEvent(self, event: QMouseEvent):  # TO ENABLE MOVEMENT FOR A MODAL (frameless) SCREEN
        self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):  # TO ENABLE MOVEMENT FOR A MODAL (frameless) SCREEN
        x = event.globalPosition().x()
        y = event.globalPosition().y()
        x_w = self.offset.x()
        y_w = self.offset.y()
        self.move(int(x - x_w), int(y - y_w))

    def dragEnterEvent(self, event):  # TO HANDLE DRAG & DROP EVENTS
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # TO HANDLE DRAG & DROP EVENTS
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):  # TO HANDLE DRAG & DROP EVENTS
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            file_paths_list = []
            for url in event.mimeData().urls():
                file_paths_list.append(str(url.toLocalFile()))
            self.setInputImages(file_paths_list)
        else:
            event.ignore()


def insertInVLayout(vlayout: QVBoxLayout, label: QLabel, max_col: int = 4):
    rows: list[QHBoxLayout] = vlayout.children()  # all the rows in the grid

    if rows:  # If there are row(s) present already
        last_row: QHBoxLayout = rows[-1]  # get the last row
        col = last_row.count()  # get the no. of cols in the last row
        if col < max_col:  # if last row NOT filled yet, fill the row
            last_row.addWidget(label)
        else:  # if last row already filled, add new row
            new_row = QHBoxLayout()
            new_row.addWidget(label)
            vlayout.addLayout(new_row)
    else:  # if no rows present in the grid, add a new row
        new_row = QHBoxLayout()
        new_row.addWidget(label)
        vlayout.addLayout(new_row)


def getPropertyAnimation(target, property_name, duration, start_value, end_value,
                         easing_curve: QEasingCurve.Type = QEasingCurve.Type.Linear):
    prop_anim = QPropertyAnimation(target, property_name)
    prop_anim.setDuration(duration)
    prop_anim.setStartValue(start_value)
    prop_anim.setEndValue(end_value)
    prop_anim.setEasingCurve(easing_curve)
    return prop_anim


def getCopyOfAnimation(old_anim: QPropertyAnimation):
    newAnim = QPropertyAnimation()
    newAnim.setTargetObject(old_anim.targetObject())
    newAnim.setDuration(old_anim.duration())
    newAnim.setPropertyName(old_anim.propertyName())
    newAnim.setEasingCurve(old_anim.easingCurve())
    newAnim.setStartValue(old_anim.startValue())
    newAnim.setEndValue(old_anim.endValue())
    return newAnim


def savePreferences(key: str, path: str):
    preferences[key] = path
    with open(evestr.PREFERENCES_PATH, 'w') as sf:
        json.dump(preferences, sf)


def setSampleImageList(folder_path: str):
    global sample_image_files, prev_folder_path
    if folder_path != prev_folder_path:
        sample_image_files = [os.path.join(folder_path, file) for file in
                              os.listdir(folder_path)]  # Get a list of all image files in the folder
    prev_folder_path = folder_path


class MovieSplashScreen(QSplashScreen):

    def __init__(self, movie:QMovie):
        movie.jumpToFrame(0)
        pixmap = QPixmap(1062,552)

        QSplashScreen.__init__(self, pixmap)
        self.movie = movie
        self.movie.frameChanged.connect(self.repaint)

        self.space_pressed = QShortcut(QKeySequence("Space"), self)
        self.space_pressed.activated.connect(self.end)
        # self.setEnabled(False)
    def end(self):
        self.movie.stop()

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        pass
    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap().scaled(1062, 552, Qt.AspectRatioMode.IgnoreAspectRatio)
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)


if __name__ == "__main__":

    app = QApplication(sys.argv)  # Initializing the QApp
    splash_movie = QMovie(evestr.SPLASH_GIF)
    splash = MovieSplashScreen(splash_movie)
    splash.show()
    splash.movie.start()

    # start = time.time() and time.time() < start + 8

    while splash_movie.state() == QMovie.MovieState.Running:
        app.processEvents()

    sample_image_files: list[str] = []
    prev_folder_path = ""

    preferences = {
        evestr.KEY_SAVE_FOLDER: evestr.CWD,
        evestr.KEY_IMAGE_FOLDER: evestr.CWD,
    }
    if os.path.exists(evestr.PREFERENCES_PATH):
        with open(evestr.PREFERENCES_PATH, 'r') as save_file:
            loaded_data: dict = json.load(save_file)
            if (evestr.KEY_IMAGE_FOLDER in loaded_data) and os.path.exists(loaded_data[evestr.KEY_IMAGE_FOLDER]):
                preferences[evestr.KEY_IMAGE_FOLDER] = loaded_data[evestr.KEY_IMAGE_FOLDER]
            if (evestr.KEY_SAVE_FOLDER in loaded_data) and os.path.exists(loaded_data[evestr.KEY_SAVE_FOLDER]):
                preferences[evestr.KEY_SAVE_FOLDER] = loaded_data[evestr.KEY_SAVE_FOLDER]

    ask_eve_qa_dict: dict = {}
    if os.path.exists(evestr.QUESTION_FILE_PATH):
        with open(evestr.QUESTION_FILE_PATH, 'r') as qaf:
            ask_eve_qa_dict = json.load(qaf)

    main_screen = MainScreen()  # Initializing an instance of the main window
    width = 1062
    height = 552
    main_screen.resize(width,height)  # Resizing the main window
    main_screen.move(width // 4, height // 4)
    main_screen.show()  # Making the main window visible on the screen

    splash.finish(main_screen)
    try:
        sys.exit(app.exec())
    except Exception:
        print("Exiting")

"""
TESTING IN DETAIL:
each image "animation" = 500 (thinking) + 500 (vertical) + 1000 (horizontal) + 1500 (thinking) = 3.5 seconds

FORMAT
x images = ( <batch_one_size> ) <batch_one_time> + ( <batch_two_size> ) <batch_two_time> = <TOTAL seconds for whole input> ( <number of image animations played before skip option could appear> )

BATCH-WISE TIME for 
100 images = (8) 2 + (92) 14 = 16.550696399994195 seconds (takes 4 animations) 

200 images = (12) 2 + (188) 29 = 31.939508200011915 seconds (takes 8 animations)

300 images = (16) 3 + (284) 44 = 47.487630899995565 seconds (takes 12 animations)

400 images = (20) 3 + (380) 58 = 61.87205879999965 seconds (takes 16 animations)

500 images = (24) 4 + (476) 73 = 77.00025279998954 seconds (takes 20 animations) 

600 images = (28) 5 + (572) 88 = 92.50088459999824 seconds (takes 25 animations) 

700 images = (32) 5 + (668) 102 = 107.25364779999654 seconds (takes 28 animations) 

800 images = (36) 6 + (764) 117 = 123.76628969999729 seconds (takes 33 animations) 

900 images = (40) 6 + (860) 138 = 145.29074969999783 seconds (takes 39 animations) 

1000 images = (44) 7 + (956) 145 = 152.52098320001096 seconds (takes 41 animations)

1500 images = (64) 10 + (1436) 219  = 228.9155643999984 seconds (takes 62 animations) 

2000 images = (84) 13 + (1916) 291 = 304.189726800003 seconds (takes 82 animations)



TESTING SUMMARY: 

100 images takes 16 seconds 

200 images takes 31 seconds

300 images takes 47 seconds 

400 images takes 61 seconds 

500 images takes 77 seconds 

600 images takes 92 seconds

700 images takes 107 seconds 

800 images takes 123 seconds

900 images takes 145 seconds

1000 images takes 152 seconds 

1500 images takes 228 seconds 

2000 images takes 304 seconds 
"""
