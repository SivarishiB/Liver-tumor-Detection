# Towards Accurate Liver Tumor Diagnosis: A Flask Implementation of Deep Learning Models

## Project Overview
This project focuses on leveraging deep learning for early and accurate liver tumor diagnosis using CT scans. It integrates three advanced deep learning models—InceptionV3, Xception, and DenseNet121—through a user-friendly Flask web application. The system enables healthcare professionals to upload CT images and receive real-time diagnostic predictions, aiding in efficient and reliable medical decision-making.

## Features
- **Deep Learning Models**: Utilizes InceptionV3, Xception, and DenseNet121 for high-accuracy tumor classification.
- **Flask Web Application**: Provides an intuitive interface for healthcare professionals.
- **Transfer Learning**: Fine-tunes pre-trained models for improved accuracy on liver CT scan datasets.
- **Data Augmentation**: Enhances model generalization using techniques like rotation, translation, and zooming.
- **Performance Evaluation**: Assesses accuracy, precision, recall, and F1-score for robust validation.

## Technologies Used
- **Programming Language**: Python
- **Frameworks**: Flask, TensorFlow, Keras
- **Libraries**: OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn
- **Deployment**: Flask-based web application

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SivarishiB/liver-tumor-diagnosis.git
   cd liver-tumor-diagnosis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the web application at:
   ```
   http://127.0.0.1:5000
   ```

## Dataset
This project uses a dataset of liver CT scans, including both benign and malignant tumor cases. Preprocessing steps include resizing, normalization, and augmentation to enhance model training.

## Usage
1. Upload a CT scan image through the web interface.
2. The selected deep learning model processes the image.
3. The application provides an immediate classification result.
4. Healthcare professionals can use the output to support diagnostic decisions.

## Results
- **DenseNet121**: Achieved the best balance of accuracy and computational efficiency.
- **Xception & InceptionV3**: Provided high accuracy but required more computational resources.
- Performance metrics are evaluated using accuracy, precision, recall, and F1-score.

## Future Enhancements
- **Incorporation of segmentation models** to highlight tumor regions.
- **3D segmentation techniques** for better visualization.
- **Integration with cloud services** for remote access.

## Contributors
- **Sivarishi B.** - SRM Institute of Science and Technology ([sb9535@srmist.edu.in](mailto:sb9535@srmist.edu.in))
- **Anto Arockia Rosaline R.** - SRM Institute of Science and Technology ([antoaror@srmist.edu.in](mailto:antoaror@srmist.edu.in))

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

