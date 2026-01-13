# Fruit Ninja Auto Slicer Computer Vision Approach

This is a silly project of mine to automate the fruit slicing game "Fruit Ninja" using computer vision techniques. The program detects fruits and bombs on the screen and performs swipe actions to slice the fruits while avoiding bomb.

The current implementation uses yolov11 and dxcam for real-time object detection and screen capturing. It can be seen on `SRC/src.py`.
An older implementation using yolvo11, mss, and deepsort can be found in `SRC/LegacyCode.ipynb`.

Disclaimer: **FOR EDUCATIONAL PURPOSE ONLY! The contributors do not assume any responsibility for the use of this tool.**

$${\color{red}Warning:}$$ This program simulates mouse movements and clicks, which may interfere with your normal computer usage. Use it at your own risk.

## Features
- Real-time object detection using YOLOv11 to identify fruits and bombs.
- Automated swipe actions to slice fruits while avoiding bombs.
- Configurable parameters for safe margin, swipe offset, and time threshold.
- A really really fast and efficient implementation of both dxcam and YOLOv11.

## Run
To run the auto-slicer, ensure you have the required dependencies installed. You can install them using pip:

```bash
pip install -r requirements.txt
```

Then, execute the main script:

```bash
python SRC/src.py
```
You can adjust the parameters such as `SAFE_MARGIN`, `SWIPE_OFFSET`, and `TIME_THRESHOLD` in the script to fine-tune the slicer's performance.


## Data Collection
The dataset for training the object detection model was collected from various Roboflow public datasets. The images and annotations later were combined and preprocessed to create a comprehensive dataset for training.

Listed below are the datasets used:
- [Fruit Ninja Dataset 1](https://universe.roboflow.com/pythonai-hf-lsuae/fruit-ninja-2noom/dataset/3)
- [Fruit Ninja Dataset 2](https://universe.roboflow.com/fruitninja-pjzb7/fruit-ninja-z0rqc)
- [Fruit Ninja Dataset 3](https://universe.roboflow.com/main-x4kob/fruit-ninja-cxmlt/dataset/1)
- [Fruit Ninja Dataset 4](https://universe.roboflow.com/labeling-dsjch/fruit-ninja-z3sce/dataset/1)
- [Fruit Ninja Dataset 5](https://universe.roboflow.com/ninja-fruit/fruit-ninja-auhst/dataset/3)
- [Fruit Ninja Dataset 6](https://universe.roboflow.com/ninja-yks0e/fruit-ninja-wsjef/dataset/3)

During preprocessing, images were resized to a uniform size, and annotations were converted to the required format for training the object detection model.
The unified dataset can be found [here](https://universe.roboflow.com/your-username/fruit-ninja-unified/dataset/1).

## Model Training
The object detection model was trained using the YOLOv11 architecture from Ultralytics. The training process involved the following steps:
1. **Environment Setup**: Installed necessary libraries including Ultralytics YOLOv11, OpenCV, and others.
2. **Data Preparation**: Loaded the preprocessed dataset and split it into training and validation sets.
3. **Model Configuration**: Configured the YOLOv11 model parameters such as input size, batch size, learning rate, and number of epochs.
4. **Training**: Initiated the training process and monitored performance metrics such as loss, precision, recall, and mAP (mean Average Precision).
5. **Evaluation**: After training, the model was evaluated on the test set to assess its performance.

## Contributings
You can contribute to this project by:
- Reporting issues or bugs.
- Suggesting new features or improvements.
- Submitting pull requests with code enhancements or bug fixes.
- Sharing your own implementations or variations of the auto-slicer.