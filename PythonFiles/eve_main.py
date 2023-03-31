import sys, os
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QStackedWidget, QMainWindow, QLabel

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
    cat_row, cat_col = 0, 0
    panda_row, panda_col = 0, 0

    def __init__(self):
        super(MainScreen, self).__init__()
        self.setupUi(self)  # -> sets the imported ui
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        widget.setWindowIcon(QIcon(evestr.ICON_QRC_PATH))  # -> sets the current window's icon
        widget.setWindowTitle(evestr.MAIN_WINDOW_TITLE)  # -> sets the current window's title

        self.pushButton_importPics.clicked.connect(self.importPicsClicked)
        self.pushButton_askEve.clicked.connect(self.askEveClicked)

    def importPicsClicked(self):
        catLabel = QLabel()
        catPixmap = QPixmap("cat.jpg").scaled(50, 50, Qt.AspectRatioMode.IgnoreAspectRatio)
        catLabel.setPixmap(catPixmap)
        self.gridLayout_cat.addWidget(catLabel, self.cat_row, self.cat_col)
        self.cat_col += 1

        if self.cat_col == 5:
            self.cat_col = 0
            self.cat_row += 1

    def askEveClicked(self):
        pandaLabel = QLabel()
        pandaPixmap = QPixmap("panda.jpg").scaled(50, 50, Qt.AspectRatioMode.IgnoreAspectRatio)
        pandaLabel.setPixmap(pandaPixmap)
        self.gridLayout_dog.addWidget(pandaLabel, self.panda_row, self.panda_col)
        self.panda_col += 1

        if self.panda_col == 5:
            self.panda_col = 0
            self.panda_row += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QStackedWidget()
    main_screen = MainScreen()
    widget.addWidget(main_screen)
    widget.resize(QSize(950, 500))
    widget.show()

    try:
        sys.exit(app.exec())
    except Exception:
        print("Exiting")
