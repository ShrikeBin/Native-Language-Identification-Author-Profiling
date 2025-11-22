# Native Language Identification and Author Profiling

Project focused on recognition of individual traits based on English writing using transformers, convolutional neural networks and a combination of both.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Data](#data)
* [Training](#training)
* [Testing & SHAP](#testing-shap)
* [Usage & Requirements](#usage-requirements)
* [Results](#results)
* [Screenshots](#screenshots)
* [License](#license)

---

## Project Structure

```
.
├── App/                   # How to run it yourself in a Nutshell
├── Assets/                # Images, Screenshots
├── Testing/               # Evaluation scripts, notebooks, and test runs of models
├── Training/              # Everything related to model training
├── ACCURACIES.md          # Summary of model evaluation results
└── README.md
```

### Training Folder

```
Training/
├── DATA/                       # Raw and preprocessed datasets
├── MODELS/                     # Trained model checkpoints organized by task
├── checkTokenDistribution.py   # See if dataset is balanced in token lenght
├── deparquetize.py             # Convert parquet datasets to other formats
├── parquetize.py               # Convert CSV/text datasets to parquet
└── README.md                   # Detailed description of datasets
```

### Models (Training/MODELS)

```
MODELS/
├── age/
│   ├── DistilBERTRegressionBaseline/        # Only final regression layer trained
│   ├── DistilBERTRegressionFull/            # Fully fine-tuned
│   └── DistilBERTRegressionLoRa/            # LoRA (efficient low-rank adaptation)
├── gender/
│   ├── DistilBERTClassificationFull/        # Fully fine-tuned
│   ├── DistilBERTRegressionBaseline/        # Only final regression layer trained
│   ├── DistilBERTRegressionFull/            # Fully fine-tuned
│   ├── DistilBERTRegressionLoRA/            # LoRA adaptation
│   └── RoBERTaRegressionFull/               # Fully fine-tuned 
├── language/
│   ├── CNN/                                 # Raw CNN on RoBERTa tokenizer embeddings
│   ├── DeBERTaClassificationLoRA/           # LoRA adaptation
│   ├── DistilBERTClassificationBaseline/    # Only final classification layer trained
│   ├── DistilBERTClassificationFull/        # Fully fine-tuned
│   ├── DistilBERTClassificationFullCNNHead/ # Fully fine-tuned standard model with CNN on last layer
│   ├── DistilBERTClassificationLoRA/        # LoRA adaptation
│   ├── MpNetClassificationFull/             # Fully fine-tuned
│   ├── RoBERTaClassificationFull/           # Fully fine-tuned
│   ├── RoBERTaClassificationFullCNNHead/    # Fully fine-tuned standard model with CNN on last layer
│   ├── RoBERTaLargeClassificationFull/      # Fully fine-tuned
│   ├── RoBERTaLargeMixedCNN/                # Parallel running CNN + RoBERTaLarge fully trained, head trained on combined output
│   └── RoBERTaMixedCNN/                     # Parallel running CNN + RoBERTa fully trained, head trained on combined output
├── mbti/
│   ├── DistilBERTClassificationBaseline/    # Only final classification layer trained
│   ├── DistilBERTClassificationFull/        # Fully fine-tuned
│   └── DistilBERTClassificationLoRA/        # LoRA adaptation
└── political/
    ├── DistilBERTRegressionBaseline/        # Only final regression layer trained
    ├── DistilBERTRegressionFull/            # Fully fine-tuned
    ├── DistilBERTRegressionFullLongText/    # Unused (results very random because of cutting long articles)
    ├── DistilBERTRegressionFullRawLogits/   # Classifier head outputs without softmax
    └── DistilBERTRegressionLoRA/            # LoRA adaptation
```

Organized by prediction task:

* **Age**: Regression models using DistilBERT variants
* **Gender**: Classification and regression model
* **Language**: Various classification methods including CNNs, Transformers (DistilBERT, RoBERTa, DeBERTa, MpNet) and mixed ( CNN + RoBERTa ...)
* **MBTI**: Classification models using DistilBERT
* **Political Orientation**: Regression models using DistilBERT

---

### Testing Folder

```
Testing/
├── RUNS/                  # Results from previous runs per task
├── shap/                  # Scripts and notebooks needed for shap runs
│   ├── ModelWrapper/      # Classes for easy loading and running different models
|   |   └── ...
│   └── ... 
└── comparisons/           # Results of testing ReadyToDeploy HuggingFace models
```

* SHAP scripts allow per-model testing of actual performance
* Each subfolder under `RUNS/` corresponds to a task like `age`, `gender`, `language`, etc.

---

## Data

* Stored in `Training/DATA` in parquet format.
* Preprocessing scripts in `Training/` handle tokenization, masking, and format conversion.
* Tasks are as follows:

  * Age prediction (regression based)
  * Gender prediction (regression based)
  * Language classification
  * MBTI type classification
  * Political orientation prediction (regression based)

---

## Training

* Models are organized by task in `Training/MODELS`.
* Different variants are trained for different tasks.
* Each folder contains a `train.py` script responsible for training its model.

---

## Testing SHAP

* Use `Testing/shap/run.ipynb` to evaluate model predictions using SHAP.
* Old run results stored in `Testing/RUNS/<task>/<run>.ipynb` as reference.

---

## Usage Requirements

* `App/` folder contains the scripts neccesary to train and run recommended models when cloning this repository.
* This code allows users to play around with our project without diving to deep in the technicals.

* Overall system requirements aside from a decent GPU are as follows:
```
Python >= 3.10
PyTorch (torch) with CUDA support
CUDA toolkit
Transformers (Hugging Face)
Pandas
NumPy
scikit-learn
spaCy
SHAP
tqdm
Matplotlib (for plotting)
Jupyter Notebook (optional, for notebooks)
```

---

## Results

* Here we present a visual summary of the model evaluation across different tasks. Each plot shows performance metrics (e.g., accuracy, Root Mean Squared Error) for the corresponding task.

| Language Classification|
|------------------------|
|![Language](Assets/plots/language.png)|

| Gender Prediction | Age Prediction |
|------------------------|----------------|
|![Gender](Assets/plots/gender.png) | ![Age](Assets/plots/age.png) |

| Political Orientation | MBTI Classification |
|-----------------|-------------------|
| ![Political](Assets/plots/political.png) | ![MBTI](Assets/plots/mbti.png) |


---

## Screenshots

* Here we show some screenshots of the working prediction models:

| Screenshots (PLACEHOLDERS FOR NOW)|
|------------------------|
|![](Assets/screenshots/1.png)|
|------------------------|
|![](Assets/screenshots/2.png)|
|------------------------|
|![](Assets/screenshots/3.png)|

---

## License

Maybe in the future
