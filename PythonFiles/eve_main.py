import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from PyQt6.QtCore import QSize, Qt, QPoint, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, \
    QSequentialAnimationGroup, QTimer
from PyQt6.QtGui import QPixmap, QMouseEvent, QFont
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, \
    QSplashScreen, QDialog, QPushButton
from keras.preprocessing.image import ImageDataGenerator

import eve_strings as evestr


def convert_ui(ui_file_name: str, py_destination_path=None):
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


main_ui_file_name = f'{evestr.PARENT_DIR}/Ui/eve_ui'  # ONLY FOR DEVELOPMENT (COMMENT OUT BEFORE MAKING EXE)
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

    img_size = 225
    batch_size = 32
    testImageGenerator = ImageDataGenerator(rescale=1. / 255)  # for rescaling image

    def __init__(self):
        super(MainScreen, self).__init__()
        self.verticalLayout_dog = QVBoxLayout()
        self.verticalLayout_dog.addLayout(QHBoxLayout())

        self.verticalLayout_cat = QVBoxLayout()
        self.verticalLayout_cat.addLayout(QHBoxLayout())

        self.offset = QPoint()
        self.setupUi(self)  # -> sets the imported ui
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Eve")
        self.setAcceptDrops(True)

        self.importPicsClicked = lambda _: self.setInputImages(
            QFileDialog.getOpenFileNames(parent=self, caption="Select Images",
                                         directory=getLastFolderOpened(),
                                         filter="Images (*.jpg *.jpeg *.png)")[0])

        self.inputImagesDf: pd.DataFrame = pd.DataFrame(columns=['Filename'])
        self.pushButton_importPics.clicked.connect(self.importPicsClicked)
        self.pushButton_classify.clicked.connect(self.classifyClicked)
        self.pushButton_cat.clicked.connect(self.catClicked)
        self.pushButton_dog.clicked.connect(self.dogClicked)
        self.pushButton_close.clicked.connect(self.deleteLater)
        self.pushButton_minimize.clicked.connect(self.showMinimized)

        self.eve_model = tf.keras.models.load_model("eve_model.h5")

        input_img_anim_vertical_timing = 500
        input_img_anim_horizontal_timing = 1000

        bubble_anim_factor = 10
        bubble_anim_timing = 100

        self.label_bubble_enlarge_animation = QPropertyAnimation()
        self.label_bubble_enlarge_animation.setTargetObject(self.label_speech_bubble)
        self.label_bubble_enlarge_animation.setPropertyName(b'size')
        self.label_bubble_enlarge_animation.setDuration(bubble_anim_timing)
        self.label_bubble_enlarge_animation.setStartValue(self.label_speech_bubble.size())
        self.label_bubble_enlarge_animation.setEndValue(self.label_speech_bubble.size() + QSize(bubble_anim_factor,bubble_anim_factor))

        self.label_dialogue_move_animation = QPropertyAnimation()
        self.label_dialogue_move_animation.setTargetObject(self.label_dialogue)
        self.label_dialogue_move_animation.setPropertyName(b'pos')
        self.label_dialogue_move_animation.setDuration(bubble_anim_timing)
        self.label_dialogue_move_animation.setStartValue(self.label_dialogue.pos())
        self.label_dialogue_move_animation.setEndValue(self.label_dialogue.pos() + QPoint(bubble_anim_factor,bubble_anim_factor))

        self.label_dialogue_move_back_animation = QPropertyAnimation()
        self.label_dialogue_move_back_animation.setTargetObject(self.label_dialogue)
        self.label_dialogue_move_back_animation.setPropertyName(b'pos')
        self.label_dialogue_move_back_animation.setDuration(bubble_anim_timing)
        self.label_dialogue_move_back_animation.setStartValue(self.label_dialogue.pos() + QPoint(bubble_anim_factor,bubble_anim_factor))
        self.label_dialogue_move_back_animation.setEndValue(self.label_dialogue.pos())

        self.label_bubble_shrink_animation = QPropertyAnimation()
        self.label_bubble_shrink_animation.setTargetObject(self.label_speech_bubble)
        self.label_bubble_shrink_animation.setPropertyName(b'size')
        self.label_bubble_shrink_animation.setDuration(bubble_anim_timing)
        self.label_bubble_shrink_animation.setStartValue(self.label_speech_bubble.size() + QSize(bubble_anim_factor,bubble_anim_factor))
        self.label_bubble_shrink_animation.setEndValue(self.label_speech_bubble.size())

        self.label_bubble_dialogue_enlarge_grp = QParallelAnimationGroup()
        self.label_bubble_dialogue_enlarge_grp.addAnimation(self.label_bubble_enlarge_animation)
        self.label_bubble_dialogue_enlarge_grp.addAnimation(self.label_dialogue_move_animation)

        self.label_bubble_dialogue_shrink_grp = QParallelAnimationGroup()
        self.label_bubble_dialogue_shrink_grp.addAnimation(self.label_bubble_shrink_animation)
        self.label_bubble_dialogue_shrink_grp.addAnimation(self.label_dialogue_move_back_animation)

        self.label_bubble_pop_seq = QSequentialAnimationGroup()
        self.label_bubble_pop_seq.addAnimation(self.label_bubble_dialogue_enlarge_grp)
        self.label_bubble_pop_seq.addAnimation(self.label_bubble_dialogue_shrink_grp)

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

        self.label_animation_seq.finished.connect(self.thinkingPause)

        self.pushButton_classify.hide()

    def thinkingPause(self):
        self.label_dialogue.setFont(QFont("Gill Sans MT Condensed", 18))
        if len(self.inputImagesDf.index) > 0:
            self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.THINKING)))
            self.setDialogue(evestr.getRandomDialogue(evestr.DialogueType.THINKING))
        else:
            self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.OTHER)))
            self.setDialogue("That was fun ^_^")
        QTimer.singleShot(1500, self.classifyInputImages)

    def showSpeechBubble(self, show: bool):
        self.label_dialogue.setVisible(show)
        self.label_speech_bubble.setVisible(show)

    def setDialogue(self, dialogue:str):
        self.label_bubble_pop_seq.start()
        self.label_dialogue.setText(dialogue)

    def setInputImages(self, path_list: list[str]):
        if path_list:
            self.pushButton_importPics.hide()
            self.pushButton_classify.show()
            self.showSpeechBubble(False)

            saveLastFolderOpened(os.path.dirname(path_list[-1]))
            self.inputImagesDf = pd.concat([pd.DataFrame({'Filename': path_list}), self.inputImagesDf]).reset_index(
                drop=True)

            print("INPUT DF: ", self.inputImagesDf)
            if len(self.inputImagesDf.index) > 0:
                img = self.inputImagesDf.iloc[0]['Filename']
                resize = self.label_inputImages.height() - 50
                img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
                self.label_inputImages.setPixmap(img_pixmap)
                self.label_inputImages.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def classifyClicked(self):
        try:
            self.pushButton_classify.hide()
            self.showSpeechBubble(True)
            test_generator = self.testImageGenerator.flow_from_dataframe(self.inputImagesDf, "",
                                                                         x_col='Filename', y_col=None,
                                                                         class_mode=None, batch_size=self.batch_size,
                                                                         target_size=(self.img_size, self.img_size),
                                                                         shuffle=False)

            predict = self.eve_model.predict_generator(test_generator,
                                                       steps=np.ceil(self.inputImagesDf.shape[0] / self.batch_size))
            threshold = 0.5

            self.inputImagesDf['Category'] = np.where(predict > threshold, 1, 0)

            print("OUTPUT DF: ", self.inputImagesDf)
            self.thinkingPause()

        except Exception as e:
            QMessageBox.information(self, "ERROR4444 :((", str(e))

    def classifyInputImages(self):
        if self.label_move_animation.targetObject() is not None:
            self.label_move_animation.targetObject().deleteLater()

        if len(self.inputImagesDf.index) < 1:
            self.pushButton_importPics.show()
            return

        img_path, img_is_dog = self.inputImagesDf.iloc[0]
        self.label_eve.setPixmap(QPixmap(evestr.getRandomPose(evestr.PoseType.OTHER)))
        self.label_dialogue.setFont(QFont("Gill Sans MT Condensed", 25))
        self.setDialogue(
            evestr.getRandomDialogue(evestr.DialogueType.DOG if img_is_dog else evestr.DialogueType.CAT))
        self.inputImagesDf = self.inputImagesDf.iloc[1:, :]  # popping first row from the dataframe
        img_pixmap = QPixmap(img_path).scaled(50, 50, Qt.AspectRatioMode.IgnoreAspectRatio)

        img_label = QLabel()
        img_label.setFixedSize(50, 50)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setPixmap(img_pixmap)
        img_label.setObjectName(img_path)
        img_label.mouseDoubleClickEvent = lambda _: Image.open(img_path).show(title=img_path)

        img_label_animated = QLabel(parent=self.centralwidget)
        img_label_animated.resize(self.label_inputImages.size())
        img_label_animated.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label_animated.setPixmap(img_pixmap)
        img_label_animated.setScaledContents(True)

        x_offset = (self.label_inputImages.pos().x()) + (self.label_inputImages.size().width() // 2) - (
            0 if img_is_dog else img_label_animated.size().width())

        y_offset = (self.label_inputImages.pos().y()) - int(self.label_inputImages.size().height() / 1.5)

        def get_btn_center(btn: QPushButton):
            return QPoint(btn.pos().x() + (btn.width() // 2), btn.pos().y() + (btn.height() // 2))

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
            get_btn_center(self.pushButton_dog if img_is_dog else self.pushButton_cat))
        self.label_animation_seq.start()

        self.insertInVLayout(self.verticalLayout_dog if img_is_dog else self.verticalLayout_cat, img_label)
        if len(self.inputImagesDf.index) > 0:
            img = self.inputImagesDf.iloc[0]['Filename']
            resize = self.label_inputImages.height() - 50
            img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label_inputImages.setPixmap(img_pixmap)
        else:
            self.label_inputImages.setPixmap(QPixmap())

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

    def mousePressEvent(self, event: QMouseEvent):  # TO ENABLE DRAGGING FOR A MODAL SCREEN
        self.offset = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):  # TO ENABLE DRAGGING FOR A MODAL SCREEN
        x = event.globalPosition().x()
        y = event.globalPosition().y()
        x_w = self.offset.x()
        y_w = self.offset.y()
        self.move(int(x - x_w), int(y - y_w))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
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


def saveLastFolderOpened(folder_path: str):
    with open(evestr.PREFERENCES_PATH, 'w') as sf:
        sf.write(folder_path + "\n")


if __name__ == "__main__":

    app = QApplication(sys.argv)
    # splash_pixmap = QPixmap("cats.jpg")
    # splash = QSplashScreen(splash_pixmap)
    # splash.show()

    main_screen = MainScreen()
    main_screen.resize(QSize(1060, 552))
    main_screen.show()

    # splash.finish(main_screen)
    try:
        sys.exit(app.exec())
    except Exception:
        print("Exiting")

# 8514oem
