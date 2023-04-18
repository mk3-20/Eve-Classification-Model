import inspect
import os
import shutil
import webbrowser
from pathlib import PureWindowsPath
from datetime import datetime, timedelta

from PyQt6.QtGui import QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView
# from PyQt6.QtWebEngineCore import  QWebEngineSettings

from PyQt6.QtWidgets import QComboBox, QLineEdit, QSpinBox, QDateEdit, QTableWidget, QMessageBox, \
    QListWidget, QListWidgetItem, QHeaderView, QFileDialog, QInputDialog, QStatusBar

from PyQt6.QtCore import Qt, QDate, QDateTime, QSettings


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
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def convert_qrc(qrc_filename: str, py_destination_path=None):
    """
    {deprecated}
    Converts given .qrc in current working directory into file into .py file [USING PyQt5 pyrcc]
    returns None
    """
    try:
        if py_destination_path is None:
            py_destination_path = qrc_filename
        os.system(f"python -m PyQt5.pyrcc_main {qrc_filename}.qrc -o {py_destination_path}_rc.py")
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_combobox_text(combobox: QComboBox):
    try:
        text = combobox.currentText()
        return text
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_combobox_data(combobox: QComboBox):
    try:
        data = combobox.currentData(Qt.ItemDataRole.UserRole)
        return data
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def set_combobox_data(combobox: QComboBox, data_dict: dict):
    try:
        # data dict = {id:"value",id:"value"}
        combobox.blockSignals(True)
        combobox.clear()
        for data_id in data_dict:
            combobox.addItem(data_dict[data_id], userData=data_id)
        combobox.blockSignals(False)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_lineedit_text(lineedit: QLineEdit):
    try:
        text = lineedit.text()
        return text
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def set_current_combobox_data(combobox: QComboBox, data):
    try:
        data_index = combobox.findData(data)
        combobox.setCurrentIndex(data_index)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_widget_value(wid) -> str:
    try:
        text = ""
        # print(f"wid = {wid}")
        if isinstance(wid, QLineEdit):
            text = wid.text()
        elif isinstance(wid, QComboBox):
            text = wid.currentText()
        elif isinstance(wid, QSpinBox):
            text = str(wid.value())
        elif isinstance(wid, QDateEdit):
            date_temp = wid.date()
            text = date_temp.toString('dd-MM-yyyy')

        elif isinstance(wid, QListWidget):
            text = wid.currentItem().text()

        return text
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")
        return ""


def get_current_row(tablebox: QTableWidget):
    try:
        cur_row = tablebox.currentRow()
        return cur_row
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def ask_for_confirmation(context, question: str = None, Title: str = None) -> bool:
    try:
        title = "?"
        if Title is not None:
            title = Title
        question_to_ask = "Are you sure you want to remove selected row?"
        if question is not None:
            question_to_ask = question
        ans = QMessageBox.question(context, title, question_to_ask,
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if ans == QMessageBox.StandardButton.Yes:
            return True
        else:
            return False

    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def raise_error(context, given_error: str, title: str = 'Error!'):
    try:
        QMessageBox.critical(context, title, given_error)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def give_info(context, info: str, title: str = "!"):
    try:
        QMessageBox.information(context, title, info)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def set_date(datebox: QDateEdit, date: str):
    try:
        date_splitted = date.split("-")
        q_date = QDate(int(date_splitted[2]), int(date_splitted[1]), int(date_splitted[0]))
        datebox.setDate(q_date)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_dates_inbetween(start_datebox: QDateEdit, end_datebox: QDateEdit) -> dict:
    report_from_date = get_widget_value(start_datebox)
    report_to_date = get_widget_value(end_datebox)

    start_date = datetime.strptime(report_from_date, "%d-%m-%Y")
    end_date = datetime.strptime(report_to_date, "%d-%m-%Y")
    date_array = (start_date + timedelta(days=x) for x in
                  range(0, (end_date - start_date).days + 1))

    dates_inbetween = []
    for date_object in date_array:
        dates_inbetween.append(date_object.strftime("%d-%m-%Y"))

    return {DATE_LIST_KEY: dates_inbetween, START_DATE_KEY: report_from_date, END_DATE_KEY: report_to_date}


def get_tablecell_data(tablebox: QTableWidget, row: int, col: int):
    try:
        cell = tablebox.item(row, col)
        if cell is not None:
            data = cell.data(Qt.ItemDataRole.UserRole)
            return data
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def refresh_listbox(listbox: QListWidget, mk_db_helper, db_table_name, name_col, id_col):
    try:
        items = mk_db_helper.select_from([name_col, id_col], db_table_name,
                                         order_by_col=name_col, order_by_value="ASC")
        listbox.clear()
        for item in items:
            list_item = QListWidgetItem(str(item[0]))
            list_item.setData(Qt.ItemDataRole.UserRole, item[1])
            listbox.addItem(list_item)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def set_current_date(datebox_list: list[QDateEdit]):
    try:
        date_in_text = ""
        for datebox in datebox_list:
            datebox.setDateTime(QDateTime.currentDateTime())
            q_date = datebox.date()
            date_in_text = q_date.toString('dd-MM-yyyy')
        return date_in_text
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def maximize_table_headers(table: QTableWidget):
    try:
        table_headers = table.horizontalHeader()
        table_headers.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_webview_filename(url_or_webview):
    try:
        url = ""
        if isinstance(url_or_webview, QWebEngineView):
            webview: QWebEngineView = url_or_webview
            url = webview.url().toString()
        if isinstance(url_or_webview, str):
            url = url_or_webview

        url_temp = url.split("file:///")[1]
        filename = PureWindowsPath(fr"{url_temp}").as_posix()
        return filename
    except Exception:
        return ""


def save_file_dialog(context, url_or_webview, directory, title: str = ""):
    try:
        filename = get_webview_filename(url_or_webview)
        pdf_name = filename.split('/')[-1]
        file_dest = QFileDialog.getSaveFileName(context, f"{title}Copy {pdf_name}", directory, "PDF Files (*.pdf)")
        if len(file_dest[0]) > 1:
            shutil.copy(filename, file_dest[0])
    except Exception:
        raise_error(context, "Select a Pdf!")


def print_pdf(url_or_webview):
    try:
        filename = get_webview_filename(url_or_webview)
        webbrowser.open_new(filename)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def make_dirs(dir_list: list[str]):
    try:
        for d in dir_list:
            if not os.path.exists(d):
                os.makedirs(d)
    except Exception as e:
        function_name = inspect.currentframe().f_code.co_name
        print(f"error function name = {function_name} , error = {e}")


def get_entry_dialogbox_text(context, title: str, text: str) -> str:
    entry_text, ok = QInputDialog.getText(context, title, text)
    if ok:
        return entry_text
    else:
        return ""


def set_icon(window, icon_path: str):
    window.setWindowIcon(QIcon(icon_path))


def get_file_name(context, directory: str, caption: str = "Select a File", filter_name: str = "All Files",
                  filter_type: str = "*", initial_filter: str = '', initial_filter_type: str = ''):
    output_path = QFileDialog.getOpenFileName(parent=context, caption=caption,
                                              directory=directory, filter=f"{filter_name} (*.{filter_type})",
                                              initialFilter=f"{initial_filter} (*.{initial_filter_type})")
    try:
        return output_path[0]
    except Exception:
        return ""

def get_file_names(context, directory: str, caption: str = "Select a File", filter_name: str = "All Files",
                  filter_type: str = "*", initial_filter: str = '', initial_filter_type: str = '') -> list[str]:
    output_path = QFileDialog.getOpenFileNames(parent=context, caption=caption,
                                              directory=directory, filter=f"{filter_name} (*.{filter_type})",
                                              initialFilter=f"{initial_filter} (*.{initial_filter_type})")
    try:
        return output_path[0]
    except Exception:
        return [""]

def disable_widget(widget_list: list, stylesheet_list=None):
    if stylesheet_list is None:
        stylesheet_list = []

    if len(widget_list) != len(stylesheet_list):
        none_list = [None] * (len(widget_list) - len(stylesheet_list))
        stylesheet_list.extend(none_list)

    for widget, stylesheet in zip(widget_list, stylesheet_list):
        widget.setDisabled(True)
        if stylesheet is not None:
            widget.setStyleSheet(stylesheet)


def enable_widget(widget_list, stylesheet_list=None):
    if stylesheet_list is None:
        stylesheet_list = []
    if len(widget_list) != len(stylesheet_list):
        none_list = [None] * (len(widget_list) - len(stylesheet_list))
        stylesheet_list.extend(none_list)

    for widget, stylesheet in zip(widget_list, stylesheet_list):
        widget.setEnabled(True)
        if stylesheet is not None:
            widget.setStyleSheet(stylesheet)


def show_statusbar_msg(status_bar: QStatusBar, message: str, duration_in_msecs: int = None):
    if status_bar is not None:
        if duration_in_msecs is not None:
            status_bar.showMessage(message, duration_in_msecs)
        else:
            status_bar.showMessage(message)


def resize_font(widget_list, percentage_in_decimal):
    for widget in widget_list:
        font = widget.font()
        font.setPixelSize(widget.height() * percentage_in_decimal)
        widget.setFont(font)


class MkQSettings:
    def __init__(self, QSETTINGS: QSettings):
        self.myQSettings = QSETTINGS

    def get_Qsettings_value(self, key: str):
        settings_value = self.myQSettings.value(key)
        return settings_value

    def set_Qsettings_value(self, key: str, value: str):
        self.myQSettings.setValue(key, value)


DATE_LIST_KEY = "date_list"
START_DATE_KEY = "start_date"
END_DATE_KEY = "end_date"
