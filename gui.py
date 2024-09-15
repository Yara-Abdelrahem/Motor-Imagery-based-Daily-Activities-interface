import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QTextEdit, QComboBox ,QLabel
from EEGModels import EEG_Model_class

class EEGApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("EEG Motor Imagery Classifier")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.layout = QVBoxLayout(self.central_widget)

        # Upload button
        self.upload_button = QPushButton("Upload File")
        self.upload_button.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_button)

        # Feature selection
        self.feature_label = QLabel("Feature Selection:")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(["Decision Tree", "Logistic Regression", "RandomForest"])
        self.feature_layout = QHBoxLayout()
        self.feature_layout.addWidget(self.feature_label)
        self.feature_layout.addWidget(self.feature_combo)
        self.layout.addLayout(self.feature_layout)

        # Run button
        self.run_button = QPushButton("Run Model")
        self.run_button.clicked.connect(self.run_model)
        self.layout.addWidget(self.run_button)

        # Output text box
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.layout.addWidget(self.output_text)

    def upload_file(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Select EEG file", "", "CSV Files (*.csv)")
        if self.filepath:
            QMessageBox.information(self, "File Selected", f"File selected: {self.filepath}")

    def run_model(self):
        if not self.filepath:
            QMessageBox.warning(self, "No File", "Please upload a file first.")
            return

        feature = self.feature_combo.currentText()
        if feature == "Decision Tree":
            model = EEG_Model_class(self.filepath)
            y_pred = model.DecisionTreeClassifier()

        elif feature == "Logistic Regression":
            model = EEG_Model_class(self.filepath)
            y_pred = model.LogisticRegression_classifier()

        elif feature == "RandomForest":
            model = EEG_Model_class(self.filepath)
            y_pred = model.RandomForest_class()

        else:
            QMessageBox.critical(error_type=QMessageBox.Warning, text="Invalid Selection", detail="Please select a valid feature.")
            return

        self.output_text.clear()
        for pred in y_pred:
            self.output_text.append(f"{pred}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGApp()
    window.show()
    sys.exit(app.exec_())



