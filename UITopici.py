import sys
import webbrowser
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QFrame, QLineEdit, QHBoxLayout, QTextEdit
import yaml
import os
from dreams_mc.make_model_card import generate_modelcard

class UiTopici(QWidget):
    def __init__(self):
        super().__init__()

        self.input_path = None
        self.output_path = None

        self.setWindowTitle("ProiectTopici")
        self.setGeometry(100, 100, 700, 500)

        self.main_layout = QHBoxLayout(self)

        self.layout = QVBoxLayout()

        self.h_layout1 = QHBoxLayout()

        self.label = QLabel("Choose the path of the dataset")
        self.h_layout1.addWidget(self.label)
        self.add_content = QPushButton("Configure descriptions")
        self.add_content.clicked.connect(self.toggle_new_layout)
        self.h_layout1.addWidget(self.add_content)
        self.layout.addLayout(self.h_layout1)


        self.button1 = QPushButton("Browse")
        self.button1.clicked.connect(self.choose_path1)
        self.layout.addWidget(self.button1)

        self.label2 = QLabel("Choose the path for the output")
        self.layout.addWidget(self.label2)

        self.button2 = QPushButton("Browse")
        self.button2.clicked.connect(self.choose_path2)
        self.layout.addWidget(self.button2)



        self.label3 = QLabel("Set the number of epochs desires")
        self.layout.addWidget(self.label3)
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Enter the number of epochs, you wish to have")
        self.layout.addWidget(self.input_field)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.HLine) 
        self.line.setFrameShadow(QFrame.Sunken)  
        self.line.setFixedHeight(30)
        self.layout.addWidget(self.line)



        self.button_antrenare=QPushButton("Train the dataset")
        self.button_antrenare.clicked.connect(self.modify_config)
        self.layout.addWidget(self.button_antrenare)

        self.button_raport=QPushButton("View the report")
        self.button_raport.clicked.connect(self.view_the_report)
        self.layout.addWidget(self.button_raport)


        self.main_layout.addLayout(self.layout)


        self.right_layout = None
        self.right_layout_widgets = []
        self.text_area = None
        self.setLayout(self.main_layout)



    def choose_path1(self):
        path1 = QFileDialog.getExistingDirectory(self, "Select Directory for Path 1")
        if path1:
            self.label.setText(f"Input: {path1}")
            self.input_path = path1

    def choose_path2(self):
        path2 = QFileDialog.getExistingDirectory(self, "Select Directory for Path 2")
        if path2:
            self.label2.setText(f"Output: {path2}")
            self.output_path = path2

    def view_the_report(self):
        if(self.output_path):
            print("Generating Model Card....")
            config_file_path = './config.yaml'
            output_path = self.output_path+'/model_card.html'
            version_num = '1.0'
            generate_modelcard(config_file_path, output_path, version_num)
            url = self.output_path+"model_card"
            webbrowser.open(url)
        
        else: 
            print("Please set output path before generating the report.")
            return
    

    def modify_config(self):
        try:
            if self.right_layout:
                
                script_dir = os.path.dirname(os.path.abspath(__file__))

                config_file_path = os.path.join(script_dir, "config.yaml")

  
                with open(config_file_path, "r") as file:
                    config_data = yaml.safe_load(file)

                describe_overview = self.text_area.toPlainText()  
                config_data["describe_overview"] = describe_overview
                with open(config_file_path, "w") as file:
                    yaml.safe_dump(config_data, file)

            print(f"Updated config.yaml: {config_data}")

        except Exception as e:
            print(f"An error occurred while modifying the config file: {e}")

    def toggle_new_layout(self):

        if self.right_layout:

            for widget in self.right_layout_widgets:
                widget.deleteLater()
            self.right_layout_widgets.clear()

            self.main_layout.removeItem(self.right_layout)
            self.right_layout = None
            self.text_area= None
        else:
            self.right_layout = QVBoxLayout()

            right_label = QLabel("Describe overview")
            self.right_layout.addWidget(right_label)
            self.right_layout_widgets.append(right_label)

            self.text_area = QTextEdit()
            self.text_area.setPlaceholderText("Enter your text here...")
            self.right_layout.addWidget(self.text_area)
            self.right_layout_widgets.append(self.text_area)

            self.main_layout.addLayout(self.right_layout)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UiTopici()
    window.show()
    sys.exit(app.exec_())
