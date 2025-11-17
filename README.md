# Densenet-Quantization

This repository contains a comprehensive Python pipeline to convert TensorFlow Keras (`.h5`) models into highly optimized NVIDIA TensorRT engines. It supports conversion to **FP32**, **FP16**, and **INT8** precisions, with a strong focus on robust INT8 calibration to maintain model accuracy.

This project was developed to benchmark and accelerate model inference, demonstrating a **10x speedup** on modern GPUs with **zero accuracy loss** by converting models to TensorRT FP16.




---

## 🚀 Features

* **Keras to ONNX:** Converts `.h5` weight files into ONNX `.onnx` models.
* **ONNX to TensorRT:** Converts `.onnx` models into optimized TensorRT `.trt` engines.
* **Full Precision Support:** Generates engines in FP32, FP16, and INT8.
* **Robust INT8 Calibration:** Includes a dynamic INT8 calibrator that uses a representative dataset to minimize accuracy loss.
* **Evaluation Pipeline:** A complete, separate evaluation framework (`evaluation/`) is provided to benchmark and validate the performance (accuracy, F1, speed) of all converted models against the original TensorFlow baseline.

---

## 🛠️ Setup & Installation

### 1. Prerequisites (Crucial!)

This project relies heavily on the NVIDIA GPU computing stack. You **must** have the following installed and configured correctly:

* **NVIDIA Driver:** A recent NVIDIA driver for your GPU.
* **CUDA Toolkit:** This project was tested with **CUDA 12.x**.
* **cuDNN:** The NVIDIA cuDNN library.
* **TensorRT:** The NVIDIA TensorRT library. The Python modules (`tensorrt`, `pycuda`) are listed in `requirements.txt`.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Geridev/Densenet-Quantization.git](https://github.com/Geridev/Densenet-Quantization.git)
    cd Densenet-Quantization
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate   # On Windows
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📁 Required Data Structure

Before running any scripts, you must organize your weights and data as follows: