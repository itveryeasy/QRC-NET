# **QRC-NET: Quantum Reservoir Computing for Credit Risk Modeling**

A Python-based project combining **Quantum Reservoir Computing (QRC)** and **Machine Learning** to analyze and predict credit risks using the German Credit dataset.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Quantum Computing in QRC](#quantum-computing-in-qrc)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

---

## **Introduction**
This project implements a hybrid approach using **Quantum Reservoir Computing** to enhance feature extraction for credit risk modeling. The primary goal is to achieve better recall in identifying high-risk credit cases.

### **Why QRC?**
- Exploits quantum dynamics for efficient feature mapping.
- Enhances prediction capabilities for imbalanced datasets.

---

## **Features**
- **Quantum Circuit Construction**: Generates a customizable quantum reservoir.
- **Credit Risk Prediction**: Uses Random Forest for classification.
- **Dataset Preprocessing**: Handles categorical and numerical features seamlessly.
- **Performance Metrics**: Focuses on recall to identify risky credit cases.

---

## **Dataset**
The project uses the **German Credit dataset**:
- **Source**: UCI Machine Learning Repository.
- **Description**: 24 attributes (categorical and numerical) and a target column (`Risk`).
- **File**: Available in the `dataset` directory as `GermanCredit.csv`.

---

## **Installation**
Follow these steps to set up the project:

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/QRC-NET.git
cd QRC-NET


python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows


pip install -r requirements.txt

python main.py
