
# Precision-Recall Curve - ML



Precision-Recall Curve (PR Curve) is a graphical representation that helps us understand how well a binary classification model is doing especially when the data is imbalanced which means when one class is more dominant than other. In this curve:

- x-axis shows recall also called sensitivity or true positive rate which tells us how many actual positive cases the model correctly identified.  
- y-axis shows precision which tells us how many of the predicted positive cases were actually correct.  

Unlike the ROC curve which looks at both positives and negatives the PR curve focuses only on how well the model handles the positive class. This makes it useful when the goal is to detect important cases like fraud, diseases or spam messages.

---

## Key Concepts of Precision and Recall

Before understanding the PR curve let’s understand:

### 1. Precision

It refers to the proportion of correct positive predictions (True Positives) out of all the positive predictions made by the model i.e True Positives + False Positives. It is a measure of the accuracy of the positive predictions. The formula for Precision is:

```

Precision = True Positives / (True Positives + False Positives)

```

A high Precision means that the model makes few False Positives. This metric is especially useful when the cost of false positives is high such as email spam detection.

---

### 2. Recall

It is also known as Sensitivity or True Positive Rate where we measures the proportion of actual positive instances that were correctly identified by the model. It is the ratio of True Positives to the total actual positives i.e True Positives + False Negatives. The formula for Recall is:

```

Recall = True Positives / (True Positives + False Negatives)

```

A high Recall means the model correctly identifies most of the positive instances which is critical when False Negatives are costly like in medical diagnoses.

---

### 3. Confusion Matrix

To better understand Precision and Recall we can use a Confusion Matrix which summarizes the performance of a classifier in four essential terms:

- True Positives (TP): Correctly predicted positive instances.  
- False Positives (FP): Incorrectly predicted positive instances.  
- True Negatives (TN): Correctly predicted negative instances.  
- False Negatives (FN): Incorrectly predicted negative instances.  

---

## How Precision-Recall Curve Works

The PR curve is created by changing the decision threshold of your model and checking how the precision and recall change at each step. The threshold is the cutoff point where you decide:

- If the probability is above the threshold you predict positive.  
- If it's below you predict negative.  

By default this threshold is usually 0.5 but you can move it up or down.

---

## PR Curve vs ROC Curve

A PR curve is useful when dealing with imbalanced datasets where one class significantly outnumbers the other. In such cases the ROC curve might show overly optimistic results as it doesn’t account for class imbalance as effectively as the Precision-Recall curve.

It is desired that the algorithm should have both high precision and high recall. However most machine learning algorithms often involve a trade-off between the two. A good PR curve has greater AUC (area under the curve).

---

## When to Use PR Curve and ROC Curve

Choosing between ROC and Precision-Recall depends on the specific needs of the problem, understanding data distribution and the consequences of different types of errors.

The PR curve helps us visualize how well our model is performing across various thresholds. It provides insights into how changes in decision thresholds affect Precision and Recall. For example increasing the threshold might increase Precision i.e fewer False Positives but it could lower Recall i.e more False Negatives. The goal is to find a balance that best suits the specific needs of your application whether it’s minimizing False Positives or False Negatives.

ROC curves are suitable when the class distribution is balanced and false positives and false negatives have similar consequences. They show the trade-off between sensitivity and specificity.
<img width="1120" height="420" alt="image" src="https://github.com/user-attachments/assets/2a4e5a13-8101-4839-88ac-a137ed3ee137" />

