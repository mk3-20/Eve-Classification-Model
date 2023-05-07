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

# IMPORTS

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from PyQt6.QtCore import QSize, Qt, QPoint, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, \
    QSequentialAnimationGroup, QTimer
from PyQt6.QtGui import QPixmap, QMouseEvent, QFont, QColor, QPainter, QBrush, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QSplashScreen, QDialog, QPushButton

from keras.preprocessing.image import ImageDataGenerator

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


class ClassifiedOutputDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Classified Output")
        self.setStyleSheet("background-color:rgb(0, 255, 255);")
        self.offset = QPoint()

    def mousePressEvent(self, event: QMouseEvent):  # TO ENABLE DRAGGING FOR A MODAL SCREEN
        self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):  # TO ENABLE DRAGGING FOR A MODAL SCREEN
        x = event.globalPosition().x()
        y = event.globalPosition().y()
        x_w = self.offset.x()
        y_w = self.offset.y()
        self.move(int(x - x_w), int(y - y_w))


class MainScreen(QMainWindow, Ui_MainWindow):
    cat_row, cat_col = 0, 0
    dog_row, dog_col = 0, 0
    max_col = 4

    img_size = 225  # (for DL Model)
    batch_size = 32  # (for DL Model)
    testImageGenerator = ImageDataGenerator(rescale=1. / 255)  # for rescaling image (for DL Model)

    def __init__(self):  # Constructor
        super(MainScreen, self).__init__()  # Calling the super class's constructor
        self.setupUi(self)  # -> sets the imported ui
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)  # Making the window frameless i.e. removing the title bar
        self.setWindowTitle("Eve")  # Setting the app's title
        self.setAcceptDrops(True)  # Allowing the app to accept drag & drop events
        self.pushButton_importPics.setIcon(QIcon(evestr.IMAGE_ICON))
        self.pushButton_importMore.setIcon(QIcon(evestr.IMPORT_ICON))
        self.pushButton_resetImports.setIcon(QIcon(evestr.RESET_ICON))
        self.pushButton_collapseAccuracy.setIcon(QIcon(evestr.COLLAPSE_RIGHT_ICON))
        self.pushButton_expandAccuracy.setIcon(QIcon(evestr.COLLAPSE_LEFT_ICON))
        basket_pixmap_size = QSize(89,89)
        self.label_catBasket.setPixmap(QPixmap(evestr.CAT_BASKET_IMG).scaled(basket_pixmap_size,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation))
        self.label_dogBasket.setPixmap(QPixmap(evestr.DOG_BASKET_IMG).scaled(basket_pixmap_size,Qt.AspectRatioMode.IgnoreAspectRatio,Qt.TransformationMode.SmoothTransformation))

        self.mainwindow_enlarge_animation = getPropertyAnimation(
            target=self,
            property_name=b'size',
            duration=750,
            start_value=QSize(0, 0),
            end_value=QSize(1060, 552),
        )
        self.mainwindow_enlarge_animation.start()

        self.verticalLayout_dog = QVBoxLayout()
        self.verticalLayout_dog.addLayout(QHBoxLayout())

        self.verticalLayout_cat = QVBoxLayout()
        self.verticalLayout_cat.addLayout(QHBoxLayout())

        self.offset = QPoint()  # To make the frameless window movable
        self.inputImagesDf: pd.DataFrame = pd.DataFrame(
            columns=['Filename'])  # Dataframe that'll include the filenames and the output class (0 for cat, 1 for dog)
        self.eve_model = tf.keras.models.load_model("eve_model.h5")
        self.incorrect_classifications_list: list[str] = []

        self.importPicsClicked = lambda _: self.setInputImages(
            # Called when user wants to select images from local storage
            QFileDialog.getOpenFileNames(parent=self, caption="Select Images",
                                         directory=getLastFolderOpened(),
                                         filter="Images (*.jpg *.jpeg *.png)")[0]
        )

        # Connecting Signals (button clicks) and Slots (functions)
        self.pushButton_importPics.clicked.connect(self.importPicsClicked)
        self.pushButton_importMore.clicked.connect(self.importPicsClicked)
        self.pushButton_resetImports.clicked.connect(self.resetImportsClicked)
        self.pushButton_classify.clicked.connect(self.classifyClicked)
        self.pushButton_close.clicked.connect(self.deleteLater)
        self.pushButton_minimize.clicked.connect(self.showMinimized)
        self.pushButton_collapseAccuracy.clicked.connect(self.collapseAccuracyClicked)
        self.pushButton_expandAccuracy.clicked.connect(self.expandAccuracyClicked)

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

        btn_anim_timing = 100
        btn_anim_factor = 10

        self.btn_catdog_enlarge_animation = getPropertyAnimation(
            target=None,
            property_name=b'size',
            duration=btn_anim_timing,
            start_value=self.label_catBasket.size(),
            end_value=self.label_catBasket.size() + QSize(btn_anim_factor, btn_anim_factor)
        )

        self.btn_catdog_shrink_animation = getPropertyAnimation(
            target=None,
            property_name=b'size',
            duration=btn_anim_timing,
            start_value=self.label_catBasket.size() + QSize(btn_anim_factor, btn_anim_factor),
            end_value=self.label_catBasket.size()
        )

        self.btn_catdog_pop_seq = QSequentialAnimationGroup()
        self.btn_catdog_pop_seq.addAnimation(self.btn_catdog_enlarge_animation)
        self.btn_catdog_pop_seq.addAnimation(self.btn_catdog_shrink_animation)

        input_img_anim_vertical_timing = 500
        input_img_anim_horizontal_timing = 1000

        self.label_move_animation_vertical = QPropertyAnimation()
        self.label_move_animation_vertical.setPropertyName(b'pos')
        self.label_move_animation_vertical.setDuration(input_img_anim_vertical_timing)
        self.label_move_animation_vertical.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.label_move_animation = QPropertyAnimation()
        self.label_move_animation.setPropertyName(b'pos')
        self.label_move_animation.setDuration(input_img_anim_horizontal_timing)
        self.label_move_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.label_shrink_animation = QPropertyAnimation()
        self.label_shrink_animation.setPropertyName(b'size')
        self.label_shrink_animation.setEndValue(QSize(0, 0))
        self.label_shrink_animation.setDuration(input_img_anim_horizontal_timing)

        self.label_animation_grp = QParallelAnimationGroup()
        self.label_animation_grp.addAnimation(self.label_move_animation)
        self.label_animation_grp.addAnimation(self.label_shrink_animation)

        self.label_animation_seq = QSequentialAnimationGroup()
        self.label_animation_seq.addAnimation(self.label_move_animation_vertical)
        self.label_animation_seq.addAnimation(self.label_animation_grp)

        self.label_animation_seq.finished.connect(self.endClassificationAnimation3)

        accuracy_expand_timing = 1000
        accuracy_collapse_timing = 500
        eve_move_anim_factor = QPoint(210, 0)

        def get_expand_accuracy_anim(t, factor=eve_move_anim_factor):
            return getPropertyAnimation(target=t, property_name=b'pos',
                                        duration=accuracy_expand_timing, start_value=t.pos(),
                                        end_value=t.pos() - factor,
                                        easing_curve=QEasingCurve.Type.OutCurve)

        def get_collapse_accuracy_anim(t, factor=eve_move_anim_factor):
            return getPropertyAnimation(target=t, property_name=b'pos',
                                        duration=accuracy_collapse_timing, start_value=t.pos() - factor,
                                        end_value=t.pos(),
                                        easing_curve=QEasingCurve.Type.OutCurve)

        self.accuracy_expand_anim_move_eve = get_expand_accuracy_anim(self.label_eve)
        self.accuracy_expand_anim_move_bubble = get_expand_accuracy_anim(self.label_speech_bubble)
        self.accuracy_expand_anim_move_dialogue = get_expand_accuracy_anim(self.label_dialogue)
        self.accuracy_expand_anim_move_frame = get_expand_accuracy_anim(self.frame_accuracy,
                                                                        QPoint(self.frame_accuracy.width(), 0))

        self.accuracy_collapse_anim_move_eve = get_collapse_accuracy_anim(self.label_eve)
        self.accuracy_collapse_anim_move_bubble = get_collapse_accuracy_anim(self.label_speech_bubble)
        self.accuracy_collapse_anim_move_dialogue = get_collapse_accuracy_anim(self.label_dialogue)
        self.accuracy_collapse_anim_move_frame = get_collapse_accuracy_anim(self.frame_accuracy,
                                                                        QPoint(self.frame_accuracy.width(), 0))

        self.expand_accuracy_anim_grp = QParallelAnimationGroup()
        self.expand_accuracy_anim_grp.addAnimation(self.accuracy_expand_anim_move_eve)
        self.expand_accuracy_anim_grp.addAnimation(self.accuracy_expand_anim_move_bubble)
        self.expand_accuracy_anim_grp.addAnimation(self.accuracy_expand_anim_move_dialogue)
        self.expand_accuracy_anim_grp.addAnimation(self.accuracy_expand_anim_move_frame)
        self.expand_accuracy_anim_grp.finished.connect(self.pushButton_collapseAccuracy.show)

        self.collapse_accuracy_anim_grp = QParallelAnimationGroup()
        self.collapse_accuracy_anim_grp.addAnimation(self.accuracy_collapse_anim_move_eve)
        self.collapse_accuracy_anim_grp.addAnimation(self.accuracy_collapse_anim_move_bubble)
        self.collapse_accuracy_anim_grp.addAnimation(self.accuracy_collapse_anim_move_dialogue)
        self.collapse_accuracy_anim_grp.addAnimation(self.accuracy_collapse_anim_move_frame)
        self.collapse_accuracy_anim_grp.finished.connect(self.pushButton_expandAccuracy.show)

        # Making the classify button invisible at first
        self.pushButton_classify.hide()
        self.pushButton_importMore.hide()
        self.pushButton_resetImports.hide()
        self.pushButton_collapseAccuracy.hide()
        self.pushButton_expandAccuracy.hide()
        self.label_catBasket.hide()
        self.label_dogBasket.hide()
        self.pushButton_speedUp.hide()
        self.pushButton_slowDown.hide()
        self.pushButton_askEve.show()
        self.pushButton_credits.show()

        self.just_identified_dog = False
        self.first_image_flag = True
        self.current_batch: list[tuple[str,int]] = []

        QTimer.singleShot(3000, self.intro)


    def intro(self):
        self.setDialogue("I'm Eve :D")

    def showSpeechBubble(self, show: bool):
        self.label_dialogue.setVisible(show)
        self.label_speech_bubble.setVisible(show)

    def setDialogue(self, dialogue: str):
        self.label_bubble_pop_seq.start()
        self.label_dialogue.setText(dialogue)

    def setInputImages(self, path_list: list[str]):  # TO UPDATE THE INPUT DATAFRAME (adding more input files)
        if path_list:
            self.pushButton_importPics.hide()
            self.pushButton_importMore.show()
            self.pushButton_resetImports.show()
            self.pushButton_classify.show()
            self.pushButton_expandAccuracy.hide()
            self.showSpeechBubble(False)

            saveLastFolderOpened(os.path.dirname(path_list[-1]) + '/')
            self.inputImagesDf = pd.concat([pd.DataFrame({'Filename': path_list}), self.inputImagesDf]).reset_index(
                drop=True)

            print("INPUT DF: ", self.inputImagesDf)
            if len(self.inputImagesDf.index) > 0:
                img = self.inputImagesDf.iloc[0]['Filename']
                resize = self.label_inputImages.height() - 50
                img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
                self.label_inputImages.setPixmap(img_pixmap)
                self.label_inputImages.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resetImportsClicked(self):
        self.pushButton_importPics.show()
        self.pushButton_importMore.hide()
        self.pushButton_resetImports.hide()
        self.pushButton_classify.hide()
        self.showSpeechBubble(True)
        self.inputImagesDf = pd.DataFrame(columns=['Filename'])
        if self.current_batch:
            self.pushButton_expandAccuracy.show()
            self.setDialogue("You can import more images or check my accuracy")
        else:
            self.setDialogue("Please give me some images to work with :D")

    def classifyClicked(self):
        """
            ACTUAL Classification of the input images using DL Model and setting the output in the DATAFRAME
        """
        try:
            self.pushButton_classify.hide()
            self.pushButton_importMore.hide()
            self.pushButton_resetImports.hide()
            self.label_catBasket.show()
            self.label_dogBasket.show()
            self.pushButton_speedUp.show()
            self.pushButton_slowDown.show()
            self.pushButton_askEve.hide()
            self.pushButton_credits.hide()
            self.showSpeechBubble(True)

            test_generator = self.testImageGenerator.flow_from_dataframe(self.inputImagesDf, "",
                                                                         x_col='Filename', y_col=None,
                                                                         class_mode=None, batch_size=self.batch_size,
                                                                         target_size=(self.img_size, self.img_size),
                                                                         shuffle=False)

            predict = self.eve_model.predict(test_generator,
                                             steps=np.ceil(self.inputImagesDf.shape[0] / self.batch_size))
            threshold = 0.5  # >0.5 = 1 (Dog)  | <0.5 = 0 (Cat)

            self.inputImagesDf['Category'] = np.where(predict > threshold, 1, 0)

            print("OUTPUT DF: ", self.inputImagesDf)
            self.thinkingPauseAnimation1()

        except Exception as e:
            QMessageBox.information(self, "ERROR4444 :((", str(e))

    def collapseAccuracyClicked(self):
        self.pushButton_importPics.show()
        self.pushButton_collapseAccuracy.hide()
        self.pushButton_importPics.setDisabled(False)
        self.label_eve.setPixmap(QPixmap(evestr.getPose(evestr.EVE_HANDFOLD)))
        self.setDialogue("You can import more images or check my accuracy")
        self.collapse_accuracy_anim_grp.start()

    def expandAccuracyClicked(self):
        self.pushButton_expandAccuracy.hide()
        self.label_eve.setPixmap(QPixmap(evestr.getPose(evestr.EVE_HANDBOARD)))
        self.setDialogue("Please select the pictures which I classified incorrectly")
        self.expand_accuracy_anim_grp.start()


    def accuracyBtnClicked(self):
        pass

    def thinkingPauseAnimation1(self):
        self.label_dialogue.setFont(QFont("Gill Sans MT Condensed", 18))
        self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.THINKING)))
        self.label_dialogue.setText(evestr.getRandomDialogue(evestr.DialogueType.THINKING))
        QTimer.singleShot(1500, self.classificationAnimation2)

    def classificationAnimation2(self):
        if self.label_move_animation.targetObject() is not None:
            self.label_move_animation.targetObject().deleteLater()

        if len(self.inputImagesDf.index) < 1:
            return

        img_path, img_is_dog = self.inputImagesDf.iloc[0]
        self.current_batch.append((img_path, img_is_dog))

        self.just_identified_dog = img_is_dog
        self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.POINTING)))
        self.first_image_flag = False
        self.label_dialogue.setFont(QFont("Gill Sans MT Condensed", 25))
        self.setDialogue(
            evestr.getRandomDialogue(evestr.DialogueType.DOG if img_is_dog else evestr.DialogueType.CAT))
        self.inputImagesDf = self.inputImagesDf.iloc[1:, :]  # popping first row from the dataframe

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

        self.label_move_animation_vertical.setStartValue(img_label_animated.pos())
        self.label_move_animation_vertical.setEndValue(QPoint(x_offset, y_offset))
        self.label_move_animation_vertical.setTargetObject(img_label_animated)

        self.label_move_animation.setTargetObject(img_label_animated)
        self.label_shrink_animation.setTargetObject(img_label_animated)
        self.label_shrink_animation.setStartValue(img_label_animated.size())
        self.label_move_animation.setStartValue(QPoint(x_offset, y_offset))
        self.label_move_animation.setEndValue(
            get_basket_center(self.label_dogBasket if img_is_dog else self.label_catBasket))
        self.label_animation_seq.start()

        self.insertInVLayout(self.verticalLayout_dog if img_is_dog else self.verticalLayout_cat, img_label)
        if len(self.inputImagesDf.index) > 0:
            img = self.inputImagesDf.iloc[0]['Filename']
            resize = self.label_inputImages.height() - 50
            img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label_inputImages.setPixmap(img_pixmap)
        else:
            self.label_inputImages.setPixmap(QPixmap())

    def endClassificationAnimation3(self):
        if not self.first_image_flag:
            target_btn = self.label_dogBasket if self.just_identified_dog else self.label_catBasket
            self.btn_catdog_enlarge_animation.setTargetObject(target_btn)
            self.btn_catdog_shrink_animation.setTargetObject(target_btn)
            self.btn_catdog_pop_seq.start()

        if len(self.inputImagesDf.index) > 0:
            QTimer.singleShot(500, self.thinkingPauseAnimation1)

        else:
            self.label_eve.setPixmap(QPixmap(evestr.getPose(evestr.EVE_CELEBRATING)))
            self.setDialogue("That was fun ^_^")
            self.first_image_flag = True
            QTimer.singleShot(1500, self.accuracyAnimation4)

    def accuracyAnimation4(self):
        self.label_catBasket.hide()
        self.label_dogBasket.hide()
        self.pushButton_speedUp.hide()
        self.pushButton_slowDown.hide()
        self.pushButton_askEve.show()
        self.pushButton_credits.show()
        self.pushButton_importPics.setDisabled(True)
        self.label_dialogue.setFont(QFont("Gill Sans MT Condensed", 18))
        self.label_eve.setPixmap(QPixmap(evestr.getPose(evestr.EVE_HANDBOARD)))
        self.setDialogue("Now let's see how accurate I was..")
        self.expand_accuracy_anim_grp.start()
        self.scrollAreaWidgetContents_cats.setLayout(self.verticalLayout_cat)
        self.scrollAreaWidgetContents_dogs.setLayout(self.verticalLayout_dog)

    def insertInVLayout(self, vlayout: QVBoxLayout, label: QLabel):
        rows: list[QHBoxLayout] = vlayout.children()  # all the rows in the grid

        if rows:  # If there are row(s) present already
            last_row: QHBoxLayout = rows[-1]  # get the last row
            col = last_row.count()  # get the no. of cols in the last row
            if col < self.max_col:  # if last row NOT filled yet, fill the row
                last_row.addWidget(label)
            else:  # if last row already filled, add new row
                new_row = QHBoxLayout()
                new_row.addWidget(label)
                vlayout.addLayout(new_row)
        else:  # if no rows present in the grid, add a new row
            new_row = QHBoxLayout()
            new_row.addWidget(label)
            vlayout.addLayout(new_row)

    def catClicked(self):
        catDialog = ClassifiedOutputDialog(self)
        catDialog.setLayout(self.verticalLayout_cat)
        catDialog.resize(750, 350)
        catDialog.show()

    def dogClicked(self):
        dogDialog = ClassifiedOutputDialog(self)
        dogDialog.setLayout(self.verticalLayout_dog)
        dogDialog.resize(750, 350)
        dogDialog.show()

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


def getLastFolderOpened():
    folderPath = evestr.CWD
    if os.path.exists(evestr.PREFERENCES_PATH):
        with open(evestr.PREFERENCES_PATH) as sf:
            folderPath = sf.readline()
    return folderPath


def getPropertyAnimation(target, property_name, duration, start_value, end_value,
                         easing_curve: QEasingCurve.Type = QEasingCurve.Type.Linear):
    prop_anim = QPropertyAnimation(target, property_name)
    prop_anim.setDuration(duration)
    prop_anim.setStartValue(start_value)
    prop_anim.setEndValue(end_value)
    prop_anim.setEasingCurve(easing_curve)
    return prop_anim


def saveLastFolderOpened(folder_path: str):
    with open(evestr.PREFERENCES_PATH, 'w') as sf:
        sf.write(folder_path + "\n")


if __name__ == "__main__":

    app = QApplication(sys.argv)  # Initializing the QApp
    # splash_pixmap = QPixmap("cats.jpg")
    # splash = QSplashScreen(splash_pixmap)
    # splash.show()

    main_screen = MainScreen()  # Initializing an instance of the main window
    width = 1062
    height = 552
    # center_of_screen = QPoint(
    main_screen.resize(QSize(0, 0))  # Resizing the main window
    main_screen.move(width // 4, height // 4)
    main_screen.show()  # Making the main window visible on the screen

    # splash.finish(main_screen)
    try:
        sys.exit(app.exec())
    except Exception:
        print("Exiting")

# 8514oem
