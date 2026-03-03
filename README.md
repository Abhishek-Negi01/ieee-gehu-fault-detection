# IEEE ML Challenge - Fault Detection System

## Project Overview

This repository contains the complete solution for the IEEE ML Challenge organized by IEEE SB GEHU. The challenge focuses on binary classification for fault detection in industrial devices. The goal is to predict whether a device is operating under normal conditions or exhibiting a faulty condition based on 47 numerical features.

## Problem Statement

The dataset consists of 43776 rows and 48 columns. The first 47 columns are numerical float features named F01 to F47, and the 48th column is the target class. The classification task is defined as:

- Class 0: Device operating under normal conditions
- Class 1: Device exhibiting a faulty condition

The training data is provided in TRAIN.csv and the test data in TEST.csv. TEST.csv contains an ID column and 47 feature columns. The final submission requires FINAL.csv with two columns ID and CLASS in the exact same order as TEST.csv.

## Research Journey and Methodology

### 1. Logistic Regression

Logistic Regression was used as the first baseline algorithm. It required StandardScaler because it is sensitive to feature scale. The saga solver was chosen as it is the fastest for large datasets. class_weight balanced was used to handle class imbalance. This model served as a simple linear baseline to establish initial performance expectations.

### 2. Decision Tree

Decision Tree was the second algorithm tested. It finds the best split at each node using gini or entropy criteria. A fully grown tree overfits badly on 43776 rows. max_depth tuning was performed using GridSearchCV to find the optimal tree depth. While interpretable and fast, Decision Tree proved weak compared to ensemble methods.

### 3. Random Forest

Random Forest builds many trees where each tree sees a random subset of rows via bootstrap sampling and a random subset of features at each split. It uses the best split from that random subset. class_weight balanced was used to handle class imbalance. GridSearchCV was attempted but ran for more than 2 hours on CPU without completing because sklearn tree building is inherently slow.

GPU acceleration was attempted using cuML from RAPIDS. cuML RandomForestClassifier with bootstrap set to False and max_features set to 1.0 replicates ExtraTrees behavior on GPU. However, the default sklearn Random Forest with max_depth None actually outperformed the GPU tuned version for several reasons:

- cuML caps max_depth internally and does not truly replicate unlimited depth
- cuML does not support class_weight balanced which hurt recall
- cuML uses histogram based splitting with n_bins 256 which loses split precision compared to sklearn exact splits

The default CPU model was kept as the better result.

### 4. Extra Trees

Extra Trees is similar to Random Forest but uses completely random split thresholds instead of searching for the best split. It does not use bootstrap so all rows are used for each tree. It is faster than Random Forest on CPU because it skips the optimal split search. Extra Trees outperformed Random Forest on every metric.

Extra Trees cannot run on GPU because cuML does not have ExtraTreesClassifier. The behavior can be approximated with cuML RandomForest using bootstrap False and max_features 1.0 but native sklearn Extra Trees on CPU with n_jobs negative one was used instead and it was fast enough.

### 5. XGBoost

XGBoost builds trees sequentially where each tree corrects errors of the previous one. It has native CUDA GPU support via device cuda which works perfectly on Google Colab T4 GPU without any extra installation. scale_pos_weight was used instead of class_weight because XGBoost does not support class_weight. scale_pos_weight is calculated as count of negative class divided by count of positive class.

Manual parameter search was used instead of GridSearchCV because each configuration runs in seconds on GPU. The tuned model with n_estimators 500, max_depth 8, learning_rate 0.05, subsample 0.9 gave the best result.

### 6. LightGBM

LightGBM uses leaf-wise tree growth instead of level-wise which makes it faster and more accurate than XGBoost in many cases. The most important parameter is num_leaves not max_depth.

GPU support was attempted but failed on Colab:

- device gpu gave the error No OpenCL device found because Colab T4 does not support OpenCL
- device cuda gave the error CUDA Tree Learner was not enabled in this build because Colab's preinstalled LightGBM is not compiled with CUDA support

CPU with n_jobs negative one was used instead and it was fast enough because LightGBM is built on highly optimized C++ with histogram based splits. LightGBM Tuned with n_estimators 500, num_leaves 63, learning_rate 0.05 became the best single model overall.

### 7. CatBoost

CatBoost was tested as the seventh algorithm. GPU was attempted using task_type GPU but failed with the error CUDA driver version is insufficient for CUDA runtime version on Colab. CPU was used instead with task_type CPU and thread_count negative one. CatBoost has built-in overfitting resistance but underperformed LightGBM and XGBoost in this experiment possibly because class_weight support on CPU behaves differently from scale_pos_weight in other boosting libraries.

### 8. MLP Neural Network

TensorFlow and Keras were used for the MLP implementation. StandardScaler was mandatory because MLP is sensitive to feature scale unlike tree models. BatchNormalization, Dropout, and EarlyStopping were used to prevent overfitting. class_weight dict was passed to model.fit to handle class imbalance. ReduceLROnPlateau was used to automatically reduce learning rate when validation loss stopped improving.

Despite tuning architecture from 128 64 to 512 256 128, MLP significantly underperformed all boosting models. The reason is that neural networks need far more data and heavy feature engineering to compete with gradient boosting on tabular data. Boosting is generally superior for structured tabular data with numerical features.

### 9. Stacking Ensemble

The final approach was a Stacking Ensemble. The three best and most diverse models were chosen as base models:

- LightGBM Tuned: Chosen because it was the best single model
- XGBoost Tuned: Chosen because it was the second best single model
- Extra Trees: Chosen instead of CatBoost because diversity matters in stacking and Extra Trees uses a fundamentally different approach which is bagging with random splits versus sequential boosting

Logistic Regression was chosen as the meta learner because the base models are already very strong and a simple combiner is sufficient and less prone to overfitting.

Key configuration details:

- passthrough True was used so the meta learner also receives the original 47 features in addition to the 3 base model probabilities
- cv 5 was used so each base model trains on 4 folds and predicts on the 5th which generates out of fold predictions for the meta learner and prevents data leakage
- stack_method predict_proba was used so calibrated probabilities are passed to the meta learner

A convergence warning appeared for Logistic Regression because with passthrough True it receives 50 features which are mixed scale. Changing solver to saga broke the model completely causing F1 to drop from 0.9876 to 0.7806 because saga behaves poorly with mixed scale unscaled inputs. The fix was to keep lbfgs solver and increase max_iter to 5000. The warning is harmless and results are valid.

## Complete Results

| Model            | Accuracy | Precision | Recall | F1     | ROC-AUC | MCC    | Log Loss | TP   | FP  | TN   | FN  |
| ---------------- | -------- | --------- | ------ | ------ | ------- | ------ | -------- | ---- | --- | ---- | --- |
| RF Default       | 0.9825   | 0.9955    | 0.9607 | 0.9778 | 0.9991  | 0.9637 | 0.0982   | 3326 | 15  | 5131 | 136 |
| RF Tuned GPU     | 0.9807   | 0.9943    | 0.9575 | 0.9756 | 0.9991  | 0.9601 | 0.1050   | 3315 | 19  | 5127 | 147 |
| Extra Trees      | 0.9862   | 0.9961    | 0.9694 | 0.9826 | 0.9995  | 0.9714 | 0.0865   | 3356 | 13  | 5133 | 106 |
| XGBoost Default  | 0.9847   | 0.9860    | 0.9757 | 0.9808 | 0.9987  | 0.9681 | 0.0509   | 3378 | 48  | 5098 | 84  |
| XGBoost Tuned    | 0.9884   | 0.9904    | 0.9806 | 0.9855 | 0.9992  | 0.9758 | 0.0372   | 3395 | 33  | 5113 | 67  |
| LightGBM Default | 0.9845   | 0.9860    | 0.9754 | 0.9807 | 0.9986  | 0.9679 | 0.0512   | 3377 | 48  | 5098 | 85  |
| LightGBM Tuned   | 0.9899   | 0.9933    | 0.9815 | 0.9874 | 0.9994  | 0.9790 | 0.0354   | 3398 | 23  | 5123 | 64  |
| CatBoost Default | 0.9757   | 0.9713    | 0.9682 | 0.9698 | 0.9975  | 0.9495 | 0.0967   | 3352 | 99  | 5047 | 110 |
| CatBoost Tuned   | 0.9862   | 0.9844    | 0.9812 | 0.9828 | 0.9990  | 0.9712 | 0.0565   | 3397 | 54  | 5092 | 65  |
| MLP Default      | 0.9430   | 0.9378    | 0.9191 | 0.9284 | 0.9845  | 0.8811 | 0.1662   | 3182 | 211 | 4935 | 280 |
| MLP Tuned        | 0.9633   | 0.9575    | 0.9509 | 0.9542 | 0.9926  | 0.9236 | 0.1054   | 3292 | 146 | 5000 | 170 |
| Stacking Final   | 0.9900   | 0.9893    | 0.9858 | 0.9876 | 0.9996  | 0.9792 | 0.0276   | 3413 | 37  | 5109 | 49  |

## Final Model Selection

The final model is the Stacking Ensemble. While LightGBM Tuned had slightly higher accuracy and comparable F1 score, Stacking was chosen for critical reasons related to the fault detection domain.

In fault detection systems:

- A False Negative means a faulty device is predicted as normal and goes undetected which can cause equipment damage, safety hazards, and costly failures
- A False Positive means a normal device is flagged as faulty which only results in an unnecessary inspection

Stacking has 49 False Negatives versus 64 False Negatives for LightGBM Tuned, which means 15 fewer faulty devices are missed. Stacking also has better Log Loss at 0.0276 versus 0.0354 and better ROC-AUC at 0.9996 versus 0.9994.

In fault detection, the priority order is Recall first, then F1, then Precision. The Stacking Ensemble achieves the best balance of these metrics while minimizing the most critical error type.

## Project Structure

```
ieee-ml-fault-detection/
├── notebooks/
│   └── notebook.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── requirements.txt
├── FINAL.csv
└── README.md
```

- notebooks/notebook.ipynb: Contains all experimental code and model comparisons
- src/train.py: Training script that builds and saves the final stacking model
- src/predict.py: Prediction script that loads the model and generates FINAL.csv
- requirements.txt: Python dependencies
- FINAL.csv: Final submission file with ID and CLASS columns

## Requirements

- Python 3.12
- numpy
- pandas
- scikit-learn
- xgboost
- lightgbm

## How to Run on Google Colab

1. Open a new Colab notebook
2. Set runtime to T4 GPU via Runtime then Change runtime type then T4 GPU
3. Mount Google Drive
4. Upload TRAIN.csv and TEST.csv to the working directory
5. Install dependencies by running:

```
!pip install numpy pandas scikit-learn xgboost lightgbm
```

6. Run training by executing:

```
python src/train.py
```

7. Run prediction by executing:

```
python src/predict.py
```

8. Download FINAL.csv using the Colab files panel or files.download function

## How to Run on Local Machine

1. Clone the repository:

```
git clone <https://github.com/Abhishek-Negi01/ieee-gehu-fault-detection.git>
```

2. Navigate to the project folder:

```
cd ieee-gehu-fault-detection
```

3. Create a virtual environment:

```
python -m venv venv
```

4. Activate the virtual environment:

- On Linux and Mac:

```
source venv/bin/activate
```

- On Windows:

```
venv\Scripts\activate
```

5. Install dependencies:

```
pip install -r requirements.txt
```

6. Place TRAIN.csv and TEST.csv in the root folder

7. Run training:

```
python src/train.py
```

8. Run prediction:

```
python src/predict.py
```

9. FINAL.csv will be generated in the root folder

## Conclusion

This project demonstrates a comprehensive approach to fault detection using machine learning. Through systematic experimentation with classical algorithms, ensemble methods, gradient boosting, and neural networks, we achieved a final model with 98.58% recall and 98.76% F1 score. The Stacking Ensemble combines the strengths of LightGBM, XGBoost, and Extra Trees to minimize false negatives, which is critical for preventing undetected faults in industrial systems.
