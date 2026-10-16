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
- **Test accuracy:** 96.05%
- **Test messages:** 1,115
- **Correctly classified normal messages:** 966
- **Correctly classified spam messages:** 105

The model missed 44 spam messages in this test set, showing that further improvements are possible.

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
