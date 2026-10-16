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
6. Evaluate the predictions on unseen test data.
7. Allow users to enter custom messages for classification.

## Results

The Multinomial Naive Bayes classifier was evaluated on 1,115 unseen test messages.

- **Test accuracy:** 96.05%
- **Precision (Spam):** 1.00 (100%)
- **Recall (Spam):** 0.70 (70%)
- **F1-score (Spam):** 0.83 (83%)

### Confusion Matrix

The confusion matrix is:

| | Predicted Ham | Predicted Spam |
|---|---:|---:|
| Actual Ham | 966 | 0 |
| Actual Spam | 44 | 105 |

### Key Observations

- **966 ham messages** were correctly classified as ham.
- **105 spam messages** were correctly classified as spam.
- **44 spam messages** were incorrectly classified as ham.
- **0 ham messages** were incorrectly classified as spam.

The model achieved 96.05% accuracy and a spam precision of 100% on this test set. However, its spam recall was 70%, meaning it detected 105 of the 149 spam messages and missed 44. Improving spam recall is an important area for future work.
## How to Run

Install the required libraries:

```bash
python -m pip install pandas scikit-learn
```

Place the `SMSSpamCollection` dataset file in the same folder as `spam_classifier.py`.

Run the program from the project root:

```bash
python Task2-Spam-Classifier/spam_classifier.py
```

Enter an SMS message when prompted. Type `exit` to stop the program.

## Limitations
The classifier may miss some spam messages or incorrectly classify unfamiliar messages. Its predictions should not be treated as a guarantee that a message is safe.

## Future Improvements
- Improve spam detection recall.
- Add a simple web interface.
- Test other machine learning algorithms.
- Add more evaluation visualizations.

## Author
CodeNova Internship — Task 2
