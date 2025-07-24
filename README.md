# Wind Turbine Blade Damage Detection using Computer Vision

## Project Overview
This project leverages **Deep Learning** and **Computer Vision** techniques to automatically identify and localize damage (such as cracks, erosion, and other defects) on wind turbine blades from image data.  
The primary goal is to develop a tool that aids in **predictive maintenance**, making inspections safer, faster, and more cost-effective.

This repository contains all the source code, trained models, and documentation developed as part of a portfolio project for a **Master's degree application**.

---

## 📜 Table of Contents
- [Background and Motivation](#-background-and-motivation)
- [Tech Stack](#️-tech-stack)
- [Repository Structure](#-repository-structure)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)
- [Results](#-results)
- [Limitations and Future Work](#-limitations-and-future-work)
- [License](#-license)

---

## 🎯 Background and Motivation
The maintenance of wind turbines is a critical and expensive operation. Manual inspections are time-consuming, pose significant risks to technicians, and can lead to extended periods of turbine downtime.  
Automating this process with **Artificial Intelligence** can drastically reduce costs and hazards while enabling the early detection of faults, thereby preventing catastrophic failures and optimizing energy production.

---

## 🛠️ Tech Stack
- **Language:** Python 3.11  
- **Deep Learning Framework:** TensorFlow / PyTorch  
- **Computer Vision Library:** OpenCV  
- **Data Analysis:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn  
- **Development Environment:** Jupyter Notebooks, Visual Studio Code  

---

## 📁 Repository Structure
```
├── data/
│   ├── sample_images/      <- Contains a few sample images for testing
│   └── README_DATA.md      <- Instructions on where to download the full dataset
├── models/                 <- Stores trained model files (.h5, .pth, etc.)
├── notebooks/              <- Jupyter Notebooks for exploration and experimentation
├── src/                    <- Main source code (.py files)
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── .gitignore              <- Specifies intentionally untracked files to ignore
├── LICENSE                 <- Project license file (e.g., MIT)
├── README.md               <- This file
└── requirements.txt        <- List of Python dependencies
```

---

## 🚀 Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/wind-turbine-damage-detection.git
   cd wind-turbine-damage-detection
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage
To run a prediction on a new image, use the `predict.py` script from the `src` directory.

**Example command:**
```bash
python src/predict.py --image_path data/sample_images/blade_damage_01.jpg
```

---

## 📊 Results
This section showcases the model's performance and output.

**Sample Detection:**  
*Example of a crack detected on a turbine blade by the model.*

**Performance Metrics:**
| Metric     | Value |
|------------|-------|
| Accuracy   | 0.00  |
| Precision  | 0.00  |
| Recall     | 0.00  |
| F1-Score   | 0.00  |

---

## 🚧 Limitations and Future Work

**Current Limitations:**
- The model was trained on a dataset of limited size, which may affect its generalization to unseen data.
- Performance can be sensitive to variations in lighting conditions, image quality, and camera angles.

**Future Work:**
- **Data Augmentation:** Expand the dataset with more diverse examples of damage types and environmental conditions.
- **Model Architecture:** Experiment with more advanced architectures (e.g., YOLOv8, Mask R-CNN) to potentially improve accuracy and localization.
- **Optimization:** Optimize the model for deployment on edge devices for real-time analysis during drone inspections.

---

## 📄 License
This project is licensed under the **MIT License**. See the LICENSE file for more details.