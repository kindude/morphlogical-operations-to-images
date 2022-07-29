import os
import sys
import cv2
import numpy as np

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import *


class ChildWindow(QtWidgets.QDialog):
    def __init__(self, name, parent):
        super(ChildWindow, self).__init__(parent)
        self.kernel1 = None
        self.ui = uic.loadUi(os.path.join(os.path.dirname(__file__), "UI\\morphOperations.ui"), self)
        self.p = parent
        self.iterations_s = 0
        self.f = True
        self.form = None
        self.label_3.setText(name)
        self.Name = name
        self.comboBox.addItems(["Квадрат", "Крест", "Эллипс"])
        self.comboBox.activated[str].connect(self.onActivated)

        if name == 'Erode' or name == 'Dilate':
            self.lineEdit_3.setVisible(True)
            self.label_2.setVisible(True)
            self.f = True
        else:
            self.lineEdit_3.setVisible(False)
            self.label_2.setVisible(False)
            self.f = False
        self.pushButton.clicked.connect(self.get_data)

    def onActivated(self, text):
        self.form = text

    def get_data(self):
        if self.Name != 'Contour':
            if self.form == "Квадрат":
                self.kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                         ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
            if self.form == "Эллипс":
                self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
            if self.form == "Крест":
                self.kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                         ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
            if self.f:
                self.iterations_s = int(self.lineEdit_3.text())
            self.p.kernel, self.p.iterations, self.p.Name = self.kernel1, self.iterations_s, self.Name
            self.close()
        else:
            if self.form == "Квадрат":
                self.kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                         ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
            if self.form == "Эллипс":
                self.kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
            if self.form == "Крест":
                self.kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                         ((int(self.lineEdit.text())), int(self.lineEdit_2.text())))
            self.p.kernel_f_cnt = self.kernel1
            self.close()


class MainWindow(QtWidgets.QMainWindow):
    G_height = 3
    G_width = 3
    sigma1 = 1
    sigma2 = 1
    image_ = []

    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi(os.path.join(os.path.dirname(__file__), "UI\\main.ui"), self)
        self.set_icon()
        self.menu_ = {}
        self.getItems()
        self.kernel_f_cnt = 0
        self.connect()
        self.filename = 'media/photos/8f72bdb90d3c35736c80b032d7c6bc61.png'
        self.set_image()

    def getItems(self):
        self.menu_ = {'upload': self.down_2,
            'save': {
                'RGB': self.RGB_3,
                'Grey': self.Grey_3,
                'GreyNormalize': self.GreyNormalize_3,
                'GreyNormalizeAndFilter': self.GreyNormalizeAndFilter_3,
                'GreyLab2': self.GreyLab2_3,
                'GreyMorph':  self.GreyMorph_3,
                'Contour':  self.Contour_3
            },
            'morph': {
                'Erode': self.Erode,
                'Dilate': self.Dilate,
                'Opening': self.Opening,
                'Closing': self.Closing,
            },
            'Cont': self.Contour,
        }

    def connect(self):
        self.menu_['upload'].triggered.connect(self.open_file)
        for item in self.menu_['save'].values():
            item.triggered.connect(self.save_image)
        for item in self.menu_['morph'].values():
            item.triggered.connect(self.second_window)
        self.menu_['Cont'].triggered.connect(self.contour_params)
        self.pushButton.clicked.connect(self.OnBtnClick)

    def save_image(self, action):
        sender = self.sender()
        name = str(sender.objectName().strip('_3') + '_2')
        label = self.findChild(QtWidgets.QLabel, name)
        label.pixmap().save('media/savedImages/' + name.strip('_3') + '.png')

    def OnBtnClick(self):
        self.G_height = int(self.plainTextEdit_4.toPlainText())
        self.G_width = int(self.plainTextEdit_3.toPlainText())
        self.sigma1 = int(self.plainTextEdit_5.toPlainText())
        self.sigma2 = int(self.plainTextEdit_6.toPlainText())
        self.GreyNormalize_2.setPixmap(convert(self.norm_grey_image))
        self.normAndFilter = cv2.GaussianBlur(self.norm_grey_image, (self.G_height, self.G_width), self.sigma1,self.sigma2)
        self.normAndF = self.normAndFilter.astype(np.uint8)
        self.GreyNormalizeAndFilter_2.setPixmap(convert(self.normAndFilter))
        th3 = cv2.adaptiveThreshold(self.normAndF, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1801, 1)
        if self.kernel_f_cnt == 0:
            kernel = np.ones((5, 5), 'uint8')
            th4 = cv2.dilate(th3, kernel, cv2.BORDER_REFLECT, iterations=10)
        else:
            th4 = cv2.dilate(th3, self.kernel_f_cnt, cv2.BORDER_REFLECT, iterations=self.iterations)

        self.GreyMorph_2.setPixmap(convert(th4))
        self.GreyLab2_2.setPixmap(convert(th3))

    def open_file(self):
        self.filename = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', os.path.join(os.path.dirname(__file__), 'media\\photos'))[0])
        if self.filename:
            self.set_image()

    def set_icon(self):
        appIcon = QIcon('media/icons/icons8-пустой-фильтр-50.png')
        self.setWindowIcon(appIcon)

    def set_image(self):
        self.image_ = cv2.imread(self.filename)
        self.RGB_2.setPixmap(convert(self.image_))
        self.grey = cv2.cvtColor(self.image_, cv2.COLOR_RGB2GRAY)
        self.Grey_2.setPixmap(convert(self.grey))
        self.norm_grey_image = cv2.normalize(self.grey, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.GreyNormalize_2.setPixmap(convert(self.norm_grey_image))
        self.normAndFilter = cv2.GaussianBlur(self.norm_grey_image, (self.G_height, self.G_width), self.sigma1, self.sigma2)
        self.normAndF = self.normAndFilter.astype(np.uint8)
        self.GreyNormalizeAndFilter_2.setPixmap(convert(self.normAndFilter))
        self.th3 = cv2.adaptiveThreshold(self.normAndF, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1801, 1)
        self.GreyLab2_2.setPixmap(convert(self.th3))
        self.kernel = np.ones((5, 5), 'uint8')
        self.iterations = 1
        self.th4 = cv2.dilate(self.th3, self.kernel, cv2.BORDER_REFLECT, iterations=self.iterations)
        self.GreyMorph_2.setPixmap(convert(self.th4))

        gradient = cv2.morphologyEx(self.th4, cv2.MORPH_GRADIENT, self.kernel)
        self.Contour_2.setPixmap(convert(self.Prewitt(self.th4)))

    def updateMorphology(self, event):
        if self.Name == 'Erode':
            self.th4 = cv2.erode(self.th3, self.kernel, cv2.BORDER_REFLECT, iterations=self.iterations)
        elif self.Name == 'Dilate':
            self.th4 = cv2.dilate(self.th3, self.kernel, cv2.BORDER_REFLECT, iterations=self.iterations)
        elif self.Name == 'Opening':
            self.th4 = cv2.morphologyEx(self.th3, cv2.MORPH_OPEN, self.kernel)
        elif self.Name == 'Closing':
            self.th4 = cv2.morphologyEx(self.th3, cv2.MORPH_CLOSE, self.kernel)
        self.GreyMorph_2.setPixmap(convert(self.th4))
        gradient = cv2.morphologyEx(self.th4, cv2.MORPH_GRADIENT, self.kernel)
        self.Contour_2.setPixmap(convert(gradient))

    def updateCnt(self, event):
        gradient = cv2.morphologyEx(self.th4, cv2.MORPH_GRADIENT, self.kernel_f_cnt)
        self.Contour_2.setPixmap(convert(gradient))

    def second_window(self, action):
        sender = self.sender()
        self.child_window = ChildWindow(sender.objectName(), self)
        self.child_window.show()
        self.child_window.setFocus()
        self.child_window.closeEvent = self.updateMorphology

    def contour_params(self):
        sender = self.sender()
        self.child_window = ChildWindow(sender.objectName(), self)
        self.child_window.show()
        self.child_window.closeEvent = self.updateCnt

    def Prewitt(self, grayImage):
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
        # Turn uint8
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Prewitt


def convert(image):
    im_resize = cv2.resize(image, (500, 500))
    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    qp = QPixmap()
    qp.loadFromData(im_buf_arr)
    return qp


def main():

    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.showMaximized()
    app.exec_()


if __name__ == '__main__':
    main()
