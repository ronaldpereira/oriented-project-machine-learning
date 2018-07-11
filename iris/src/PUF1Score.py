def f1_score(true_positive, true_negative, false_positive, false_negative):
    # Calculates precision
    precision = (true_positive / (true_positive + true_negative)) if true_positive > 0 else 0

    # Calculates recall
    recall = (true_positive / (true_positive + false_negative)) if true_positive > 0 else 0

    # Calculates f1_score
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score

def calculateF1Score(predictions, answers):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for index, pred in enumerate(predictions):
        # True Positive
        if pred == answers[index] and pred == 1:
            true_positive += 1
        
        # True Negative
        elif pred == answers[index] and pred == 0:
            true_negative += 1

        # False Positive
        elif pred != answers[index] and pred == 1:
            false_positive += 1

        # False Negative
        elif pred != answers[index] and pred == 0:
            false_negative += 1

    return f1_score(true_positive, true_negative, false_positive, false_negative)
