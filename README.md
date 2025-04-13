# Soybean Counting Project

This project explores methods for counting soybeans in an image, addressing the challenge of densely packed beans that may touch or slightly overlap.

## Introduction

Accurately counting small, dense objects like soybeans in images has applications in agriculture and quality control. This project starts by applying traditional Computer Vision (CV) techniques and then extends towards a Deep Learning approach to improve accuracy and generalization capabilities.

## Approaches

### 1. Traditional Computer Vision (CV)

This method utilizes a pipeline combining several techniques:

*   **HSV Color Segmentation:** Isolates soybean pixels from the background based on their characteristic color range in the HSV color space.
*   **Morphological Operations:** Uses operations like Opening and Closing to filter noise and improve the shape of the soybean mask.
*   **Watershed Algorithm:** Separates touching or slightly overlapping soybeans based on the Distance Transform of the processed mask.
*   **Parameter Optimization (Random Search):** Automatically searches for the optimal set of parameters (HSV thresholds, morphology kernel, distance transform threshold) to maximize accuracy on the test image.

**Highlight Result:** Using parameters optimized through 5000 iterations of Random Search, this method achieved a count of **829/830** beans (~99.9% accuracy) on the original test image. However, detailed analysis revealed this high accuracy might be due to a compensation effect between over-segmentation (splitting single beans) and under-segmentation (merging touching beans or missing beans).

### 2. Deep Learning (Instance Segmentation - YOLOv8)

To address the limitations of the CV method, particularly in handling touching beans and generalizing to new images, the project includes data prepared for a Deep Learning approach:

*   **Data:** A dataset comprising **25 small image patches** cropped from the original images.
*   **Annotations:** Each soybean within these 25 images has been **meticulously labeled** with segmentation masks in the **YOLO format**. The data is provided in `soybean_yolo_dataset.zip`.
*   **Model:** The plan is to use the **YOLOv8-seg** model (or similar instance segmentation models) to train for detecting and segmenting individual soybeans.

## Dataset

*   **Original Image:** `Pic_3.jpg` (The primary test image for the CV method).
*   **YOLO Data:** `soybean_yolo_dataset.zip`
    *   Unzipping this file will create the standard YOLO directory structure (typically including `images` and `labels` folders).
    *   **Important:** You need to create a `data.yaml` file to specify the paths to the training/validation directories (for initial testing, you might use all 25 images for both train and val, or split them), the number of classes (only 1: 'soybean'), and the class name.

    Example `data.yaml` content:
    ```yaml
    train: path/to/soybean_yolo_dataset/images # Or specify a dedicated train set
    val: path/to/soybean_yolo_dataset/images   # Or specify a dedicated val set

    nc: 1
    names: ['soybean']
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <YOUR_REPOSITORY_NAME>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    venv\Scripts\activate
    ```
3.  **Install dependencies:**
    *   (Create a `requirements.txt` file if you don't have one)
    ```
    opencv-python
    numpy
    matplotlib
    tqdm
    ultralytics # YOLOv8 library
    # Add other necessary libraries
    ```
    *   Run the command:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Unzip the YOLO data:**
    ```bash
    # Use the appropriate unzip tool for your OS
    unzip soybean_yolo_dataset.zip
    ```
5.  **Create `data.yaml`:** As described in the Dataset section.

## Usage

### CV Method (HSV + Watershed)

*   **Run with optimized parameters (829 beans):**
    ```bash
    python cv_watershed_optimized.py
    ```
    *(Assuming you create this file with the best-found parameters)*

*   **Run the manual HSV threshold finder tool:**
    ```bash
    python find_hsv_tool.py
    ```
    *(Press 'q' to exit and view the values)*

*   **Run the automatic parameter finder (Random Search):**
    ```bash
    python random_search_optimizer.py
    ```
    *(Note: This can take a long time depending on the number of iterations)*

### Deep Learning Method (YOLOv8-seg)

*   **Train the model:**
    *   Ensure your `data.yaml` file is set up correctly.
    *   Run the training command from the terminal (example):
    ```bash
    yolo segment train data=path/to/data.yaml model=yolov8s-seg.pt epochs=100 imgsz=640 batch=8 name=soybean_yolov8_run1
    ```
    *(Adjust `data`, `model`, `epochs`, `imgsz`, `batch`, `name` parameters as needed)*

*   **Predict (Inference) with a trained model:**
    ```bash
    yolo segment predict model=path/to/your/best_trained_model.pt source=path/to/test_image.jpg save=True
    ```

## Results

*   **CV Method:** Achieved very high quantitative accuracy (~99.9% with 829/830 beans) on the original test image after parameter optimization. However, qualitative analysis reveals local segmentation errors (over/under-segmentation) and limited generalization potential.
*   **Deep Learning Method:** This approach is expected to improve segmentation quality and perform better on diverse, unseen images after being fully trained on the labeled dataset. Training and evaluation results will be updated.

## Contributing

Contributions are welcome! Please feel free to create a Pull Request or open an Issue to discuss potential improvements or report bugs.

## License

[MIT License](LICENSE)
*(You should also create a LICENSE file containing the MIT license text)*
