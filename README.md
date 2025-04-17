# 8P361 Project AI for Medical Image Analysis

## Locally Trainable Few-Shot Classifier with Grad-CAM to Flag Potentially Missed Lymph Node Metastases in Breast Cancer Patients
Deep learning models trained on heterogeneous datasets may suffer from over-generalization when deployed locally on data subjected to uniform preprocessing. In the context of histological image analysis, this can lead to misclassifications of metastases, resulting in sub-optimal treatment decisions. This work presents a few-shot learning binary classifier of metastases presence in the sentinel lymph nodes tissue preparations of breast cancer patients, incorporating self-supervised feature extraction using masked autoencoders. This approach enables local histological laboratories to train their custom feature extractor using unlabeled dataset subjected to their own preprocessing protocol. The features learned through this self-supervised method are then utilized in transfer learning for the few-shot classifier, which significantly reduces the number of manually annotated samples required for the classifier's training, while still surpassing the baseline performance set by a panel of human histopathologists. To demonstrate that the classifier relies on the image features, and it is robust to potentially noisy labels, the model was optimized during the training to heavily penalize false negative classifications and focus especially on identifying potentially missed (false positive) metastases in the samples. Gradient Class Activation Mapping (Grad-CAM) flavored heatmaps applied to the final convolutional layer of the feature extractor visually highlight potential metastases in the images of clinically negative samples. Furthermore, a PCam Analyzer graphical user interface is provided, enabling clinicians to inspect the samples with potentially overlooked metastases sorted by model's prediction confidence for a second opinion guided by Grad-CAM visualisation.

### Methods

![Model's architecture](assets/8p361_project2.drawio.png)

#### Installing dependencies
```
python3 -m pip install -r requirements.txt
```

#### Masked autoencoder training
```
cd few_shot_classifier
python3 -m feature_extractor.masked_encoder
```

#### Generating average input subsets for few-shot classifier
```
cd few_shot_classifier
python3 -m cross_validation.cross_validation
```

#### Few-Shot classifier training
```
cd few_shot_classifier
python3 -m few_shot_training.training
```

#### Latent space plotter
```
cd few_shot_classifier
python3 -m latent_space_plotter.latent_space_plotter
```

#### ROC curve plotter
```
cd few_shot_classifier
python3 -m roc_curve_plotter.roc_curve_plotter
```

#### Generating false positive classifications
```
cd pcam_analyzer/scripts
python3 generate_false_positives.py
```

### Results

| Classifier       | Encoder    | AUC  | Kaggle AUC |
|------------------|------------|------|------------|
| Full classifier  | trainable  | 0.99 | 0.95       |
| Full classifier  | frozen     | 0.82 | 0.81       |
| Few-Shot 1000    | frozen     | 0.84 | 0.84       |
| Few-Shot 1000    | trainable  | 0.89 | 0.87       |
| Few-Shot 100     | frozen     | 0.83 | 0.82       |
| Few-Shot 100     | trainable  | 0.70 | 0.60       |
| Few-Shot 32      | frozen     | 0.82 | 0.84       |
| Few-Shot 32      | trainable  | 0.69 | 0.60       |
| Few-Shot 10      | frozen     | 0.79 | 0.80       |
| Few-Shot 10      | trainable  | 0.61 | 0.52       |
| Few-Shot 5       | frozen     | 0.76 | 0.73       |
| Few-Shot 5       | trainable  | 0.45 | 0.42       |

![ROC bechmark frozen encoder](assets/ROC_benchmark.png)

![ROC benchmark trainable encoder](assets/ROC_benchmark_trainable.png)

### PCam Analyzer GUI
PCam Analyzer graphical user interface was built in Python on the DASH framework. The Grad-
CAM, xGRAD-CAM, High resolution CAM and ScoreCAM variations are provided for comparison as well as the option
for user to select convolutional layer for which the Grad-CAM is generated.

#### Running PCam Analyzer
```
cd pcam_analyzer
python3 app.py
```
After initialization, the app shall be available at http://127.0.0.1:8050/.


![PCam Analyzer GUI](assets/pcam_gui.png)

