# LIAR Dataset Analysis README

## Data Summary

This analysis uses the LIAR dataset, a collection of over 12,000 manually labeled short statements from POLITIFACT.COM. Each statement is labeled with one of six fine-grained labels: 'pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', and 'true'. The dataset includes metadata about the speaker, their title, state, party, and the context in which the statement was made, as well as some numerical features related to the speaker's history of truthfulness.

The dataset is split into training (10240 statements) and validation (1284 statements) sets.

For the binary classification task, the labels were simplified into 'TRUE' ('true', 'mostly-true') and 'FALSE' ('half-true', 'barely-true', 'false', 'pants-fire'). The binary distribution in the training data is approximately 64.5% FALSE and 35.5% TRUE.

## Provenance

The LIAR dataset was introduced in the paper "LIAR: A Benchmark Dataset for Fake News Detection" by Wang et al. (2017). The data was collected from PolitiFact.com.

## License

The LIAR dataset is available under the [MIT License](https://github.com/Tariq60/LIAR-PLUS/blob/master/LICENSE).

## Ethical Statements

*   **Bias:** Like any dataset derived from real-world sources, the LIAR dataset may contain biases reflecting the source (PolitiFact) and the political landscape at the time of data collection. The distribution of labels, speakers, and parties could influence model performance and potentially perpetuate existing biases.
*   **Misinformation Propagation:** While intended for research in fake news detection, the dataset itself contains false and misleading statements. Care must be taken in how models trained on this data are deployed to avoid inadvertently spreading misinformation.
*   **Privacy:** The dataset contains names of individuals (speakers) and context information, which should be handled responsibly, although the data is publicly available from the source.

## Data Dictionary

The dataset contains the following columns:

*   **id:** Unique identifier for the statement (e.g., '2635.json').
*   **label:** The original six-class label assigned by PolitiFact ('pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true').
*   **statement:** The text of the statement being evaluated.
*   **speaker:** The name of the person who made the statement.
*   **title:** The title or position of the speaker.
*   **state:** The state associated with the speaker or statement.
*   **party:** The political party of the speaker.
*   **num1:** Number of times the speaker was rated 'pants-fire'.
*   **num2:** Number of times the speaker was rated 'false'.
*   **num3:** Number of times the speaker was rated 'barely-true'.
*   **num4:** Number of times the speaker was rated 'half-true'.
*   **num5:** Number of times the speaker was rated 'mostly-true'.
*   **context:** The context in which the statement was made (e.g., 'a mailer', 'a floor speech').
*   **label_enc:** Numerical encoding of the multi-class 'label'.
*   **label_binary:** Simplified binary label ('TRUE' or 'FALSE').

## Explanatory Plots

The notebook includes several explanatory plots:

1.  **Confusion Matrix - Baseline BERT Model:** This heatmap shows the performance of the initial BERT-only model on the binary classification task, indicating how many statements from each true class were predicted into each predicted class.

2.  **Confusion Matrix - Combined BERT + Metadata Model:** This heatmap shows the performance of the model combining BERT embeddings with metadata features on the binary classification task, illustrating the impact of adding metadata on classification outcomes.

3.  **ROC Curves:** This plot compares the Receiver Operating Characteristic (ROC) curves for the baseline BERT model and the combined model, showing their trade-off between true positive rate and false positive rate at various thresholds. The Area Under the Curve (AUC) provides a single metric to compare the overall performance of the models, indicating the combined model performs slightly better.

4.  **Prediction Probability Distribution:** This histogram shows the distribution of predicted probabilities for the 'TRUE' class for both actual FALSE and actual TRUE statements according to the combined model. An ideal model would show clear separation between the two distributions.