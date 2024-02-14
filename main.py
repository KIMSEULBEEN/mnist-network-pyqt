import sys
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QPixmap
from PyQt6 import uic
import MnistNet
import pyqtgraph as pg

form_main = uic.loadUiType('main.ui')[0]

class WindowClass(QMainWindow, form_main):
    def __init__(self):
        super( ).__init__( )
        self.setupUi(self)
        self.setWindowTitle("MnistNet Test Program")
        self.btn_a.clicked.connect(self.fnc_a)
        self.mnistnet = MnistNet.Net()

        self.widget_graph.plot(list(range(10)), [0] * 10)
        # self.set_bar_graph(list([0] * 10))

    def fnc_a(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File')
        filename = str(filename[0])

        # When you selected a file
        if len(filename) > 0:
            prediction, predictions = self.mnistnet.test_image(filename)
            self.label_number.setPixmap(QPixmap(filename))
            self.label_prediction.setText(str(prediction))
            # self.set_bar_graph(predictions)

            self.widget_graph.clear()


            x = range(10)
            y = predictions
            bargraph = pg.BarGraphItem(x0 = 0, y = x, height=0.6, width = y)
            # bargraph = pg.BarGraphItem(x = x, y = y, height=0.6, width=x)
            self.widget_graph.getAxis('left').setTicks([[(i, v) for i, v in enumerate(list(map(str, range(10))))]])
            self.widget_graph.addItem(bargraph)


            # self.widget_graph.plot(list(range(10)), predictions)

    def set_bar_graph(self, predictions):
        # creating a plot window
        plot = pg.plot()

        # create list for y-axis
        # y1 = [5, 5, 7, 10, 3, 8, 9, 1, 6, 2]
        y1 = predictions

        # create horizontal list i.e x-axis
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # with width = 0.6
        # with bar colors = green
        bargraph = pg.BarGraphItem(x=x, height=y1, width=0.6, brush='g')

        # add item to plot window
        # adding bargraph item to the plot window
        plot.addItem(bargraph)

        layout = QGridLayout()
        layout.addWidget(plot)

        self.widget_graph.setLayout(layout)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WindowClass()
    window.show()
    app.exec( )
    print('hello')