# SMS Spam Message Classifier

## Project Overview

This project uses machine learning to classify SMS messages as **Spam** or **Ham (Not Spam)**. It uses Natural Language Processing (NLP) techniques to analyze message text and predict its category.

## Objectives

- Identify spam and legitimate SMS messages.
- Apply text preprocessing and feature extraction.
- Train a machine learning classification model.
- Evaluate the model using accuracy, precision, recall, and F1-score.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository, containing 5,572 labeled messages.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Methodology

1. Load the SMS dataset.
2. Separate message text from labels.
3. Split the data into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate predictions on unseen test data.
7. Allow users to enter custom messages for classification.

## Results

The Multinomial Naive Bayes classifier was evaluated on 1,115 unseen test messages.

- **Test accuracy:** 97.04%
- **Precision (Spam):** 1.00 (100%)
- **Recall (Spam):** 0.78 (78%)
- **F1-score (Spam):** 0.88 (88%)

### Confusion Matrix

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| Actual Ham | 966 | 0 |
| Actual Spam | 33 | 116 |

### Key Observations

- **966 ham messages** were correctly classified as ham.
- **116 spam messages** were correctly classified as spam.
- **33 spam messages** were incorrectly classified as ham.
- **0 ham messages** were incorrectly classified as spam.

The model achieved 97.04% accuracy and 100% spam precision on this test set. Its spam recall was 78%, meaning it detected 116 of the 149 spam messages and missed 33. Improving spam recall remains an important area for future work.

The evaluation results are based on the specified dataset split and may vary if the dataset or model configuration changes.

## How to Run

### Prerequisites

Install Python and the required libraries.

```bash
python -m pip install pandas matplotlib seaborn scikit-learn
```

### Dataset Setup

Place the `SMSSpamCollection` dataset file in the same folder as `spam_classifier.py`.

### Run the Program

Open a terminal in the project folder and execute:

```bash
python spam_classifier.py
```

The program trains the classifier, displays the evaluation results, saves `evaluation_report.txt` and `confusion_matrix.png`, and allows users to test custom SMS messages.

Enter an SMS message when prompted. Type `exit` to stop the program.

## Limitations

- The classifier may miss some spam messages or incorrectly classify unfamiliar messages.
- Its predictions should not be treated as a guarantee that a message is safe.
- The model's performance depends on the dataset and its characteristics.

## Future Improvements

- Improve spam detection recall.
- Compare other machine learning algorithms.
- Add a simple web interface.
- Test the classifier on more recent SMS messages.
- Add more evaluation visualizations.

## Author

CodeNova Internship — Task 2
