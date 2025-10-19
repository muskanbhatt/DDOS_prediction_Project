from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"=== {model_name} ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Report:\n", report)
    return acc

def save_predictions(preds, filename='results/prediction_output.csv'):
    df = pd.DataFrame(preds, columns=['Predicted'])
    df.to_csv(filename, index=False)

def plot_accuracy_comparison(acc_dict):
    names = list(acc_dict.keys())
    values = list(acc_dict.values())
    plt.bar(names, values, color=['red', 'green', 'blue'])
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.show()
