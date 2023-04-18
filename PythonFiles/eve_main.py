import random
import sys, os
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import QApplication, QStackedWidget, QMainWindow, QLabel, QDialog, QFileDialog, QGridLayout, \
    QVBoxLayout, QHBoxLayout
from PIL import Image

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


main_ui_file_name = f'{evestr.PARENT_DIR}/Ui/eve_ui'
convert_ui(main_ui_file_name, "eve_ui")

from eve_ui import Ui_MainWindow


class MainScreen(QMainWindow, Ui_MainWindow):
    max_col = 4

    insert_in_cat = True

    def __init__(self):
        super(MainScreen, self).__init__()
        self.setupUi(self)  # -> sets the imported ui
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        self.inputImagePaths: list[str] = []

        self.pushButton_importPics.clicked.connect(self.importPicsClicked)
        self.pushButton_classify.clicked.connect(self.classifyClicked)

        self.shuffle_timer = QTimer()
        self.shuffle_timer.timeout.connect(self.shuffleOutput)

    def importPicsClicked(self):
        self.inputImagePaths.extend(QFileDialog.getOpenFileNames(parent=self, caption="Select Images",
                                                                 directory=evestr.PARENT_DIR,
                                                                 filter="Images (*.jpg *.jpeg *.png)")[0])
        if self.inputImagePaths:
            img = self.inputImagePaths[-1]
            resize = self.label_inputImages.height() - 50
            img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label_inputImages.setPixmap(img_pixmap)
            self.label_inputImages.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def classifyClicked(self):
        self.shuffle_timer.start(200)

    def label_clicked(self, label:QLabel):
        print(label.objectName())

    def shuffleOutput(self):
        if not self.inputImagePaths:
            self.shuffle_timer.stop()
            return

        img_path = self.inputImagePaths.pop()
        img_pixmap = QPixmap(img_path).scaled(50, 50, Qt.AspectRatioMode.IgnoreAspectRatio)

        img_label = QLabel()
        img_label.setFixedSize(50, 50)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_label.setPixmap(img_pixmap)
        img_label.setObjectName(img_path)
        img_label.mouseDoubleClickEvent = lambda _:Image.open(img_path).show(title=img_path)

        # insert_in_cat = True
        insert_in_cat = random.randint(0, 1)
        self.insertInVLayout(self.verticalLayout_cat if insert_in_cat else self.verticalLayout_dog, img_label)

        if self.inputImagePaths:
            img = self.inputImagePaths[-1]
            resize = self.label_inputImages.height() - 50
            img_pixmap = QPixmap(img).scaled(resize, resize, Qt.AspectRatioMode.IgnoreAspectRatio)
            self.label_inputImages.setPixmap(img_pixmap)
        else:
            self.label_inputImages.setPixmap(QPixmap())

    def insertInVLayout(self, vlayout: QVBoxLayout, label: QLabel):
        rows: list[QHBoxLayout] = vlayout.children()

        if rows:
            last_row: QHBoxLayout = rows[-1]
            col = last_row.count()
            if col < self.max_col:
                last_row.addWidget(label)
            else:
                new_row = QHBoxLayout()
                new_row.addWidget(label)
                vlayout.addLayout(new_row)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_screen = MainScreen()
    main_screen.resize(QSize(950, 500))
    main_screen.show()

    try:
        sys.exit(app.exec())
    except Exception:
        print("Exiting")
