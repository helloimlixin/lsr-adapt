import numpy as np
import evaluate
import logging

logger = logging.getLogger(__name__)


def get_metrics_for_task(task_name):
    """
    Load appropriate evaluation metrics for a given GLUE task.

    Args:
        task_name (str): Name of the GLUE task.

    Returns:
        dict: Dictionary of metric objects for the task.
    """
    # All tasks need accuracy
    metrics = {"accuracy": evaluate.load("accuracy")}

    # Add task-specific metrics
    if task_name.lower() in ["mrpc", "qqp"]:
        # Paraphrase tasks - add F1
        metrics["f1"] = evaluate.load("f1")
    elif task_name.lower() in ["stsb"]:
        # Regression task - add Pearson and Spearman correlation
        metrics["pearson"] = evaluate.load("pearsonr")
        metrics["spearman"] = evaluate.load("spearmanr")
    elif task_name.lower() in ["cola"]:
        # Linguistic acceptability - add Matthews correlation
        metrics["matthews"] = evaluate.load("matthews_correlation")

    logger.info(f"Loaded metrics for {task_name}: {list(metrics.keys())}")
    return metrics


def compute_metrics(eval_pred, task_name="mrpc"):
    """
    Compute evaluation metrics for model predictions.

    Args:
        eval_pred: Tuple of (predictions, labels).
        task_name (str): GLUE task name for task-specific metrics.

    Returns:
        dict: Dictionary of metric values.
    """
    metrics = get_metrics_for_task(task_name)
    logits, labels = eval_pred

    # Get predictions (argmax for classification, raw values for regression)
    if task_name.lower() == "stsb":
        # Regression task
        predictions = logits.squeeze()
    else:
        # Classification task
        predictions = np.argmax(logits, axis=-1)

    # Compute all metrics
    results = {}
    for metric_name, metric in metrics.items():
        if metric_name in ["pearson", "spearman"]:
            # Correlation metrics
            result = metric.compute(predictions=predictions, references=labels)
            results[metric_name] = result[metric_name]
        elif metric_name == "f1":
            # F1 score (use binary for MRPC/QQP)
            result = metric.compute(predictions=predictions, references=labels, average="binary")
            results[metric_name] = result["f1"]
        else:
            # Other metrics (accuracy, matthews)
            result = metric.compute(predictions=predictions, references=labels)
            results[metric_name] = result[metric_name]

    return results