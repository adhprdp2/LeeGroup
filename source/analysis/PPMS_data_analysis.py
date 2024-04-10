# -*- coding: utf-8 -*-
"""
#Code to load, plot, analyze, and manipulate PPMS data and save plots/analyzed data.
#Authors: Evan Telford (ejt2133@columbia.edu) and Christie Koay (csk2172@columbia.edu).
#Latest-update: March 24 2023
"""
##############################################################################################################################################################
#load pertinent packages
import pandas as pan
import numpy as np
import scipy as sc
import scipy.signal
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

#sets the default font style for matplotlib plots
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 6}
matplotlib.rc('font', **font)
matplotlib.use('Qt5Agg')
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit,
    QDial,
    QDoubleSpinBox,
    QFontComboBox,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QStackedLayout,
    QFileDialog,
    QWidget,
    QGroupBox,
    QTabWidget,
    QToolButton,
    QStyle,
    QInputDialog,
    QDesktopWidget
)
##############################################################################################################################################################
#define the GUI class that creates the parent window for all tabs
class App_big(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PPMS Data Analysis'
        self.left = 0
        self.top = 0
        self.width = 2000
        self.height = 1000
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #Make TableWidget
        self.table_widget = MyTableWidget_App()
        self.table_widget.tabs.addTab(App(),"Plot 1")
        # Add tabs to widget
        self.table_widget.layout.addWidget(self.table_widget.tabs)
        self.table_widget.setLayout(self.table_widget.layout)
        self.setCentralWidget(self.table_widget)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        self.show()
        self.raise_()
##############################################################################################################################################################
#defines the GUI class that creates tabs with Main_Window children and contains the canvas plot
class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.dataplot=MplCanvas(self,width=4,height=4,dpi=300)
        #Make TableWidget
        self.table_widget_main = MyTableWidget_Main(self.dataplot)
        # Add tabs
        self.main=MainWindow(self.dataplot)
        self.toolbar=NavigationToolbar(self.dataplot,self)
        self.table_widget_main.tabs.addTab(self.main, "Trace 1")
        #dataplot
        self.table_widget_main.layout.addWidget(self.dataplot,1,9,9,9)
        self.table_widget_main.layout.addWidget(self.toolbar,0,9,1,9)
        # Add tabs to widget
        self.table_widget_main.layout.addWidget(self.table_widget_main.tabs,0,0,10,5)
        self.table_widget_main.setLayout(self.table_widget_main.layout)
        self.setCentralWidget(self.table_widget_main)
##############################################################################################################################################################
#define the class that creates the plot canvas
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        for axis in ['top','bottom','left','right']:
            self.axes.spines[axis].set_linewidth(0.5)
        self.axes.xaxis.set_tick_params(width=0.5)
        self.axes.yaxis.set_tick_params(width=0.5)
        super(MplCanvas, self).__init__(self.fig)
##############################################################################################################################################################
#defines the class that is in charge of making new tabs for the big App                
class MyTableWidget_App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout(self)
        # Initialize tab screen
        self.tabs = QTabWidget(self, movable=True, tabsClosable=True)
        self.tabs.tabCloseRequested.connect(self.tabs.removeTab)
        self.tabs.tabBarDoubleClicked.connect(lambda: self.tabs.setTabText(self.tabs.currentIndex(), self.changeTabName()))
        button = QToolButton()
        button.setToolTip('Add New Plot')
        button.clicked.connect(self.addNewTab)
        button.setIcon(self.style().standardIcon(
            QStyle.SP_FileDialogNewFolder))
        self.tabs.setCornerWidget(button, Qt.TopRightCorner)
      
    def changeTabName(self):
        name, done1 = QInputDialog.getText(self, 'Input Dialog','Enter Tab Name:')
        return name
        
    def addNewTab(self):
        text = 'Plot %d' % (self.tabs.count() + 1)
        self.tabs.addTab(App(), text)
##############################################################################################################################################################        
#defines the class that is in charge of making new tabs for Main_Window
class MyTableWidget_Main(QWidget):
    
    def __init__(self, plot):
        super().__init__()
        self.layout = QGridLayout(self)
        self.dataplot=plot
        # Initialize tab screen
        self.tabs = QTabWidget(self, movable=True, tabsClosable=True)
        self.tabs.tabCloseRequested.connect(self.tabs.removeTab)
        self.tabs.tabBarDoubleClicked.connect(lambda: self.tabs.setTabText(self.tabs.currentIndex(), self.changeTabName()))
        button = QToolButton()
        button.setToolTip('Add New Trace')
        button.clicked.connect(self.addNewTab)
        button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        self.tabs.setCornerWidget(button, Qt.TopRightCorner)
        
    def changeTabName(self):
        name, done1 = QInputDialog.getText(self, 'Input Dialog','Enter Tab Name:')
        return name
    
    def addNewTab(self):
        text = 'Trace %d' % (self.tabs.count() + 1)
        self.tabs.addTab(MainWindow(self.dataplot), text)
##############################################################################################################################################################
#defines the main class that's responsible for loading/analyzing/plotting data
class MainWindow(QMainWindow):

    def __init__(self, plot):
        super().__init__() #needed for initialization
        self.main_layout = QGridLayout() #create grid layout of entire window
        self.setWindowTitle("PPMS Data Analysis") #creates GUI title
        self.filename=[] #sets temp variable for the file name
        #create group boxes
        self.load_GUI()
        self.available_data()
        self.plot_or_print()
        self.create_analysis_box()
        self.create_save_box()
        self.dataplot=plot
        # self.toolbar=NavigationToolbar(self.dataplot,self)
        self.create_plot_analysis_box()
        self.common_data_analyses()
        self.clearplot_GUI()
        #Add all widgets to the overall layout
        self.main_layout.addWidget(self.LoadGUI,0,0,1,2)
        self.main_layout.addWidget(self.AvailableData, 0,2,1,1)
        self.main_layout.addWidget(self.PlotPrint, 1,0,1,2)
        self.main_layout.addWidget(self.cleardata, 2,0,1,2)
        self.main_layout.addWidget(self.CreateAnalysisBox,3,0,2,2)
        self.main_layout.addWidget(self.SaveBox,5,0,1,3)
        self.main_layout.addWidget(self.PlotAnalysisBox,1,2,3,1)
        self.main_layout.addWidget(self.CommonDataAnalysisBox,4,2,1,1)
        self.widget=QWidget()
        self.widget.setLayout(self.main_layout)
        self.setCentralWidget(self.widget) #self is a replacement to object that you will later call
        
    def clearplot_GUI(self):
        layout = QGridLayout() #creates local layout
        self.cleardata = QGroupBox('Clear Graph') #creates local box 
        l2_button=QPushButton('Clear') #creates load data push button
        l2_button.clicked.connect(lambda: self.dataplot.axes.cla())
        l2_button.clicked.connect(lambda: self.dataplot.draw())
        layout.addWidget(l2_button) 
        self.cleardata.setLayout(layout) #sets layout
        
    def getfile(self): #opens window to find and select files
        fname=QFileDialog.getOpenFileName(self, 'Open file')
        self.filename.setText(fname[0])
        
    def savefile(self): #saves data file
        dir_name=QFileDialog.getExistingDirectory(self)
        self.data.to_csv(dir_name+'/'+self.savefilename.text()+'.txt',header=self.data.columns.values,index=None, sep=',')
        line ='[Data]'
        with open(dir_name+'/'+self.savefilename.text()+'.txt', 'r+') as file: 
            file_data = file.read() 
            file.seek(0, 0) 
            file.write(line + '\n' + file_data) 
    
    def array_round(self,array,decimals):
        new_array=np.array([])
        for i,f in enumerate(array):
            new_array=np.hstack([new_array,float(np.format_float_scientific(f,decimals))])
        return new_array
        
    def load_data(self,name): #loads any quantum design data file
        self.xaxis.clear()
        self.yaxis.clear()
        self.variable_box.clear()
        self.variable_box_2.clear()
        word='[Data]'
        with open(name,encoding='latin1') as fp:
            lines=fp.readlines()
            for line in lines:
                if line.find(word) != -1:
                    start_line=lines.index(line)
                    break
        file=pan.read_csv(name, header=start_line+1,sep=',',encoding='latin1',skip_blank_lines=False)
        headers=list(file.columns.values)
        self.xaxis.addItems(headers)
        self.yaxis.addItems(headers)
        self.variable_box.addItems(headers)
        self.variable_box_2.addItems(headers)
        self.data=pan.DataFrame(file)
    
    def update_plot(self,x,y):
        self.dataplot.axes.set_xlabel(x)
        self.dataplot.axes.set_ylabel(y)
        self.dataplot.axes.plot(self.data[x],self.data[y],label=y+' vs. '+x)
        self.dataplot.fig.tight_layout()
        self.dataplot.axes.tick_params(direction="in")
        self.dataplot.axes.yaxis.set_ticks_position('both')
        self.dataplot.axes.xaxis.set_ticks_position('both')
        self.dataplot.axes.grid(linewidth=0.25,color='k',linestyle='--')
        self.dataplot.axes.legend(loc='upper right', fontsize=3)
        self.dataplot.draw()
    
    def smooth_data(self):
        smooth_y=sc.signal.savgol_filter(self.data[self.yaxis.currentText()],int(self.binning_window.text()),polyorder=1)
        x=self.data[self.xaxis.currentText()]
        if self.smooth_check.isChecked():
            new_data=self.yaxis.currentText()+',smoothed,bins='+self.binning_window.text()
            self.data[new_data]=smooth_y
            headers=list(self.data.columns.values)
            xaxis_temp=self.xaxis.currentText()
            yaxis_temp=self.yaxis.currentText()
            vb_temp=self.variable_box.currentText()
            vb2_temp=self.variable_box_2.currentText()
            self.xaxis.clear()
            self.yaxis.clear()
            self.variable_box.clear()
            self.variable_box_2.clear()
            self.xaxis.addItems(headers)
            self.yaxis.addItems(headers)
            self.variable_box.addItems(headers)
            self.variable_box_2.addItems(headers)
            self.xaxis.setCurrentText(xaxis_temp)
            self.yaxis.setCurrentText(yaxis_temp)
            self.yaxis.setCurrentText(new_data)
            self.variable_box.setCurrentText(vb_temp)
            self.variable_box_2.setCurrentText(vb2_temp)
        else:
            pass
        self.dataplot.axes.plot(x,smooth_y,label=self.yaxis.currentText()+' vs. '+self.xaxis.currentText()+' smoothed')
        self.dataplot.axes.set_xlabel(self.xaxis.currentText())
        self.dataplot.axes.set_ylabel(self.yaxis.currentText())
        self.dataplot.fig.tight_layout()
        self.dataplot.axes.tick_params(direction="in")
        self.dataplot.axes.yaxis.set_ticks_position('both')
        self.dataplot.axes.xaxis.set_ticks_position('both')
        self.dataplot.axes.grid(linewidth=0.25,color='k',linestyle='--')
        self.dataplot.axes.legend(loc='upper right', fontsize=3)
        self.dataplot.draw()
        
    def symmetrize_data(self):
        x=self.data[self.xaxis.currentText()]
        y=self.data[self.yaxis.currentText()]
        upper_bound=np.min([np.abs(np.max(x)),np.abs(np.min(x))])
        x_symm=np.linspace(0,upper_bound,len(x))
        fit=sc.interpolate.interp1d(x,y,'linear')
        y_symm=(fit(+1*x_symm)+fit(-1*x_symm))/2
        x_symm_name=self.xaxis.currentText()+'_symmetrized'
        y_symm_name=self.yaxis.currentText()+'_symmetrized'
        self.data[x_symm_name]=x_symm
        self.data[y_symm_name]=y_symm
        headers=list(self.data.columns.values)
        vb_temp=self.variable_box.currentText()
        vb2_temp=self.variable_box_2.currentText()
        self.xaxis.clear()
        self.yaxis.clear()
        self.variable_box.clear()
        self.variable_box_2.clear()
        self.xaxis.addItems(headers)
        self.yaxis.addItems(headers)
        self.variable_box.addItems(headers)
        self.variable_box_2.addItems(headers)
        self.xaxis.setCurrentText(x_symm_name)
        self.yaxis.setCurrentText(y_symm_name)
        self.variable_box.setCurrentText(vb_temp)
        self.variable_box_2.setCurrentText(vb2_temp)
        #plot the data
        self.dataplot.axes.plot(x_symm,y_symm,label=self.yaxis.currentText()+' vs. '+self.xaxis.currentText())
        self.dataplot.axes.set_xlabel(self.xaxis.currentText())
        self.dataplot.axes.set_ylabel(self.yaxis.currentText())
        self.dataplot.fig.tight_layout()
        self.dataplot.axes.tick_params(direction="in")
        self.dataplot.axes.yaxis.set_ticks_position('both')
        self.dataplot.axes.xaxis.set_ticks_position('both')
        self.dataplot.axes.grid(linewidth=0.25,color='k',linestyle='--')
        self.dataplot.axes.legend(loc='upper right', fontsize=3)
        self.dataplot.draw()
        
    def antisymmetrize_data(self):
        x=self.data[self.xaxis.currentText()]
        y=self.data[self.yaxis.currentText()]
        upper_bound=np.min([np.abs(np.max(x)),np.abs(np.min(x))])
        x_symm=np.linspace(0,upper_bound,len(x))
        fit=sc.interpolate.interp1d(x,y,'linear')
        y_symm=(fit(+1*x_symm)-fit(-1*x_symm))/2
        x_symm_name=self.xaxis.currentText()+'_antisymmetrized'
        y_symm_name=self.yaxis.currentText()+'_antisymmetrized'
        self.data[x_symm_name]=x_symm
        self.data[y_symm_name]=y_symm
        headers=list(self.data.columns.values)
        vb_temp=self.variable_box.currentText()
        vb2_temp=self.variable_box_2.currentText()
        self.xaxis.clear()
        self.yaxis.clear()
        self.variable_box.clear()
        self.variable_box_2.clear()
        self.xaxis.addItems(headers)
        self.yaxis.addItems(headers)
        self.variable_box.addItems(headers)
        self.variable_box_2.addItems(headers)
        self.xaxis.setCurrentText(x_symm_name)
        self.yaxis.setCurrentText(y_symm_name)
        self.variable_box.setCurrentText(vb_temp)
        self.variable_box_2.setCurrentText(vb2_temp)
        #plot the data
        self.dataplot.axes.plot(x_symm,y_symm,label=self.yaxis.currentText()+' vs. '+self.xaxis.currentText())
        self.dataplot.axes.set_xlabel(self.xaxis.currentText())
        self.dataplot.axes.set_ylabel(self.yaxis.currentText())
        self.dataplot.fig.tight_layout()
        self.dataplot.axes.tick_params(direction="in")
        self.dataplot.axes.yaxis.set_ticks_position('both')
        self.dataplot.axes.xaxis.set_ticks_position('both')
        self.dataplot.axes.grid(linewidth=0.25,color='k',linestyle='--')
        self.dataplot.axes.legend(loc='upper right', fontsize=3)
        self.dataplot.draw()
        
    def fitdata(self):
        if self.lower_fit_limit.text():
            lb=float(self.lower_fit_limit.text()) #import lower fit limit
        else:
            lb=np.min(self.data[self.xaxis.currentText()])
        if self.upper_fit_limit.text():
            up=float(self.upper_fit_limit.text()) #import lower fit limit
        else:
            up=np.max(self.data[self.xaxis.currentText()])
        if self.fit_order.text():
            fo=int(self.fit_order.text()) #import fit order
        else:
            fo=int(1)
        x=self.data[self.xaxis.currentText()]
        y=self.data[self.yaxis.currentText()]
        ind=np.where(np.abs(x-(up+lb)/2) < np.abs((up-lb)/2))
        x_temp=x[ind[0]]
        y_temp=y[ind[0]]
        y_temp_2=y_temp[~np.isnan(y_temp)]
        x_temp_2=x_temp[~np.isnan(y_temp)]
        self.fit,self.cov=np.polyfit(x_temp_2, y_temp_2, fo, cov=True)
        self.fitfunc=np.poly1d(self.fit)
        #redraw the plot
        self.dataplot.axes.plot(self.data[self.xaxis.currentText()],self.data[self.yaxis.currentText()],label=self.yaxis.currentText()+' vs. '+self.xaxis.currentText())
        self.dataplot.axes.plot(np.linspace(float(lb),float(up),1000),self.fitfunc(np.linspace(float(lb),float(up),1000)),'--',label='Fit: '+str(self.array_round(self.fit,3))+' $\pm$ '+str(self.array_round(np.sqrt(np.diag(self.cov)),3)))
        self.dataplot.axes.set_xlabel(self.xaxis.currentText())
        self.dataplot.axes.set_ylabel(self.yaxis.currentText())
        self.dataplot.fig.tight_layout()
        self.dataplot.axes.tick_params(direction="in")
        self.dataplot.axes.yaxis.set_ticks_position('both')
        self.dataplot.axes.xaxis.set_ticks_position('both')
        self.dataplot.axes.grid(linewidth=0.25,color='k',linestyle='--')
        self.dataplot.axes.legend(loc='upper right', fontsize=3)
        self.dataplot.draw()
        
    def create_new_variable(self):
        var_name=self.variable_box.currentText()
        var_name_2=self.variable_box_2.currentText()
        new_var_name=self.new_var_name.text()
        code=self.analysis_box.text()
        data=self.data[var_name]
        data_2=self.data[var_name_2]
        temp_data=eval(code.replace('x','data').replace('y','data_2')) 
        self.data[new_var_name]=temp_data
        headers=list(self.data.columns.values)
        self.xaxis.clear()
        self.yaxis.clear()
        self.variable_box.clear()
        self.variable_box_2.clear()
        self.xaxis.addItems(headers)
        self.yaxis.addItems(headers)
        self.variable_box.addItems(headers)
        self.variable_box_2.addItems(headers)
    
    def load_GUI(self): #creates the load data GUI
        layout = QGridLayout()
        self.LoadGUI = QGroupBox('Load PPMS Data') 
        #Create new widget to load data
        self.filename = QLineEdit()
        l2_button=QPushButton('Load Data')
        l2_button.clicked.connect(self.getfile)
        l2_button.clicked.connect(lambda: self.load_data(self.filename.text()))
        #adds widgets
        layout.addWidget(self.filename, 0,1) 
        layout.addWidget(l2_button, 0,0)
        self.LoadGUI.setLayout(layout)
        
    def available_data(self):
        layout = QGridLayout()
        self.AvailableData = QGroupBox('Available Data')
        self.xlabel= QLabel("x-axis")
        self.ylabel= QLabel("y-axis")
        self.xaxis=QComboBox()
        self.yaxis=QComboBox()
        #adds widgets to local box
        layout.addWidget(self.xlabel, 0,0) 
        layout.addWidget(self.ylabel, 0,1)
        layout.addWidget(self.xaxis,1,0)
        layout.addWidget(self.yaxis,1,1) 
        self.AvailableData.setLayout(layout)
        
    def plot_or_print(self):
        layout = QGridLayout()
        self.PlotPrint = QGroupBox('Plot Data') 
        l3_button=QPushButton('Plot Data')
        l3_button.clicked.connect(lambda: self.update_plot(self.xaxis.currentText(),self.yaxis.currentText())) 
        layout.addWidget(l3_button, 0,0) 
        self.PlotPrint.setLayout(layout) 

    def create_analysis_box(self):
        layout = QGridLayout()
        self.CreateAnalysisBox = QGroupBox('Create New Data Columns') 
        variable= QLabel("Variable (x)")
        self.variable_box=QComboBox()
        variable4= QLabel("Variable 2 (y)")
        self.variable_box_2=QComboBox()
        variable2= QLabel("Analysis Code")
        self.analysis_box = QLineEdit()
        variable3= QLabel("New Variable Name")
        self.new_var_name = QLineEdit()
        l_button=QPushButton('Create Column')
        l_button.clicked.connect(lambda: self.create_new_variable())
        layout.addWidget(variable, 0,0) 
        layout.addWidget(self.variable_box, 1,0)
        layout.addWidget(variable4, 2,0) 
        layout.addWidget(self.variable_box_2, 3,0)
        layout.addWidget(variable2,4,0)
        layout.addWidget(self.analysis_box, 5,0)
        layout.addWidget(variable3, 6,0)
        layout.addWidget(self.new_var_name,7,0)
        layout.addWidget(l_button,8,0) 
        self.CreateAnalysisBox.setLayout(layout)
        
    def create_save_box(self):
        layout = QGridLayout() 
        self.SaveBox = QGroupBox('Save Data')
        #Create new widget to save data
        self.savefilename = QLineEdit()
        #Crates Save Data button functions
        l2_button=QPushButton('Save Data')
        l2_button.clicked.connect(self.savefile)
        #adds widgets
        layout.addWidget(self.savefilename, 0,1) 
        layout.addWidget(l2_button, 0,0)
        self.SaveBox.setLayout(layout)
        
    def create_plot_analysis_box(self):
        layout=QGridLayout()
        self.PlotAnalysisBox = QGroupBox('Fitting Plot Data')
        variable= QLabel("Lower Limit") 
        self.lower_fit_limit = QLineEdit()
        variable2= QLabel("Upper Limit") 
        self.upper_fit_limit = QLineEdit()
        variable3= QLabel("Polynomial Fit Order") 
        self.fit_order = QLineEdit()
        l_button=QPushButton('Fit Data') 
        l_button.clicked.connect(lambda: self.fitdata()) 
        layout.addWidget(variable, 0,0,1,1) 
        layout.addWidget(self.lower_fit_limit, 1,0,1,1)
        layout.addWidget(variable2, 2,0,1,1) 
        layout.addWidget(self.upper_fit_limit, 3,0,1,1)
        layout.addWidget(variable3, 4,0,1,2) 
        layout.addWidget(self.fit_order, 5,0,1,2)
        layout.addWidget(l_button, 6,0,1,2)
        self.PlotAnalysisBox.setLayout(layout)
    
    def common_data_analyses(self):
        layout=QGridLayout()
        self.CommonDataAnalysisBox = QGroupBox('Modify Data')
        variable= QLabel("Smooth Data Binning Window") 
        self.binning_window = QLineEdit() 
        self.smooth_check=QCheckBox('Save?')
        l3_button=QPushButton('Smooth Data')
        l3_button.clicked.connect(lambda: self.smooth_data())
        variable2= QLabel("Hall Analysis") 
        l_button=QPushButton('Symmetrize Data')
        l2_button=QPushButton('Antisymmetrize Data') 
        l_button.clicked.connect(lambda: self.symmetrize_data()) 
        l2_button.clicked.connect(lambda: self.antisymmetrize_data()) 
        layout.addWidget(variable, 0,0,1,1)
        layout.addWidget(self.smooth_check,0,1,1,1)
        layout.addWidget(self.binning_window, 1,0,1,2)
        layout.addWidget(l3_button, 2,0,1,2)
        layout.addWidget(variable2, 3,0,1,2) 
        layout.addWidget(l_button, 4,0,1,1) 
        layout.addWidget(l2_button, 4,1,1,1) 
        self.CommonDataAnalysisBox.setLayout(layout)
###############################################################################        
#This is the actual code to run the gui       
app = QApplication(sys.argv)
window = App_big()
window.show()
app.exec()
