# Supplementary Index

This document describes the supplementary materials developed, how to install and run locally, and where to obtain the required dataset.

---

## Install Guide

1. **Download the full project that includes the BTXRD dataset from the GitHub repository**

   * GitHub: **\[GitHub Link]**

2. **Or download dataset from FigShare and add to project files**

   * BTXRD dataset FigShare Link: [https://figshare.com/articles/dataset/A\_Radiograph\_Dataset\_for\_the\_Classification\_Localization\_and\_Segmentation\_of\_Primary\_Bone\_Tumors/27865398?file=50653575](https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398?file=50653575)
   * Place the downloaded BTXRD folder within the projects dataset folder: /dataset/BTXRD/

3. **Please use Python 3.11**

4. **To install the requirements please run `pip install -r requirements.txt`**

---

## Links

1. **BTXRD Figshare**: [https://figshare.com/articles/dataset/A\_Radiograph\_Dataset\_for\_the\_Classification\_Localization\_and\_Segmentation\_of\_Primary\_Bone\_Tumors/27865398?file=50653575](https://figshare.com/articles/dataset/A_Radiograph_Dataset_for_the_Classification_Localization_and_Segmentation_of_Primary_Bone_Tumors/27865398?file=50653575)

2. **Full project GitHub repository**: **\[GitHub Link]**

---

## Project Content Description

### 1) Dataset

* Contains the BTXRD dataset folder
* May contain the BTXRD_YOLO_MultiClass folder for the Yolov8 model local Training

### 2) Dual Model Pipeline

* `logic.py`: Preprocessing logic used by the dual model pipeline.
* `dual_model_pipeline.py`: Core visualisation for the system, including dual model agreement. Please run this file for the full functionality of the work developed in this project.

### 3) Evaluation

* Dual Model Evaluation:

  * `dual_model_pipeline_evaluation.py`: Evaluation for the dual model pipeline.
  * `dual_model_pipeline_results.txt`: Output results for the evaluation of the dual model pipeline.

* Model evaluation folders: **YOLOv8**, **MobileNetV3**, **EfficientNet‑B3**, **ConvNeXt‑Tiny**.

  * `model_evaluation.py`: Evaluation for overall model performance.

  * `model_evaluation_anatomy.py`: Evaluation for performance per anatomical site.

  * `model_evaluation_xai.py`: Evaluation of Grad-CAM, Grad-CAM++, Layer-CAM XAI outputs.

  * Corresponding `_results.txt`: Contain the results for the performed evaluation per model.

* Exploratory Data Analysis: <br>
An Exploratory Data Analysis was conducted to evaluate the BTXRD dataset.


### 4) Example Output

* Example outputs showcasing different scenarios using the dual model pipeline.

* Highlights model classifications, segmentations, XAI Grad‑CAM overlays, and dual model agreement logic outcomes.

### 5) Models

* Models trained using Google Colab:

  * `ConvNeXt-Tiny.pt`, `EfficientNet-B3.pt`, `MobileNet-V3.pt`, `YOLOv8.pt`.


### 6) Training

* **Google Colab Training**: .ipynb notebooks for all four models trained on google colab including model training outputs.

* **Local Training**: .ipynb notebooks with local paths in order to allow for training to be reproduced locally.

* **YOLOv8 dataset conversion** — `create_yolo_dataset.py` converts the BTXRD dataset into the corresponding YOLO format for training YOLOv8. The  dataset is generated in the `dataset/BTXRD_YOLO_MultiClass` folder. 

In order to perform the Yolov8 model training locally, please first run the `create_yolo_dataset.py` code to generate the new BTXRD_YOLO_MultiClass dataset locally.

### 7) Dissertation.mp4

* A video summarising the work developed during the project, highlighting its core functionality and results.

### 8) requirements.txt

* Python requirements to reproduce the environment and run the code locally.

* Please use Python 3.11

* To install the requirements please run `pip install -r requirements.txt`

### 9) BTXRD.pdf

* BTXRD Dataset description document.

### 10) README

* Instructions and guidance on the dissertation project and its supplementary materials.

## References

BTXRD: <br>
Yao, S., Huang, Y., Wang, X. et al. A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors. Sci Data 12, 88 (2025). https://doi.org/10.1038/s41597-024-04311-y

GRAD-CAM: <br>
Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

GRAD-CAM++:<br>
Chattopadhay, A., Sarkar, A., Howlader, P. and Balasubramanian, V.N., 2018, March. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. In 2018 IEEE winter conference on applications of computer vision (WACV) (pp. 839-847). IEEE.

Layer-CAM: <br>
Jiang, P.T., Zhang, C.B., Hou, Q., Cheng, M.M. and Wei, Y., 2021. Layercam: Exploring hierarchical class activation maps for localization. IEEE transactions on image processing, 30, pp.5875-5888.

EfficientNet-B3: <br>
Tan, M. and Le, Q., 2019, May. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.

MobileNetV3: <br>
Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V. and Le, Q.V., 2019. Searching for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 1314-1324).

ConvNeXt-Tiny: <br>
Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T. and Xie, S., 2022. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 11976-11986).

Yolov8: <br>
Jocher, G.; Chaurasia, A.; Qiu, J. Ultralytics yolov8. https://github.com/ultralytics/ultralytics (2023).

Albumentations:<br>
Buslaev, A., Iglovikov, V.I., Khvedchenya, E., Parinov, A., Druzhinin, M. and Kalinin, A.A., 2020. Albumentations: fast and flexible image augmentations. Information, 11(2), p.125.

PyTorch:<br>
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. and Desmaison, A., 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32.

scikit-learn:<br>
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, pp.2825-2830.

timm: <br>
Wightman, R. (2019– ). PyTorch Image Models (timm), v1.2.2. Zenodo. https://doi.org/10.5281/zenodo.4414861