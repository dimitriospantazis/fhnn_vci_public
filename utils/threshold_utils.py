import os 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from utils.constants_utils import FIVE_PERCENT_THRESHOLD, SIX_PERCENT_THRESHOLD, SEVEN_PERCENT_THRESHOLD, EIGHT_PERCENT_THRESHOLD, COLORS, THRESHOLDS_TO_EDGE_FRACTIONS
def collect_metrics_multiple_with_thresholds(date):
    """
    Collects the Final Test Loss, ROC AUC, and Test Average Precision from multiple file runs
    in the date directory
    """
    average_precisions = []
    roc_aucs = []
    final_test_losses = []
    accuracies = []
    thresholds = []
    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    
    for log_num in os.listdir(date_dir):
        average_precision, roc_auc, final_test_loss, accuracy, threshold = \
            collect_metrics_from_log_file_with_threshold(os.path.join(date_dir, log_num, "log.txt"))
        average_precisions.append(average_precision)
        roc_aucs.append(roc_auc)
        final_test_losses.append(final_test_loss)
        accuracies.append(accuracy)
        thresholds.append(threshold)
    return average_precisions, roc_aucs, final_test_losses, accuracies, thresholds


def collect_metrics_from_log_file_with_threshold(log_file_dir):
    with open(log_file_dir) as log_file:
        log_file_lines = log_file.readlines()
        for log_file_line in log_file_lines:
            if "INFO:root:Using PLV Threshold" in log_file_line:
                # Index up to -1 to eliminate \n character
                threshold = float(log_file_line.split(" ")[-1][ : -1])
                break

        final_line = log_file_lines[-9]
        if "Early stopping" in final_line: 
            final_line = log_file_lines[-10]
        if "INFO:root:Test Epoch:" not in final_line:
            final_line = log_file_lines[-11]
        words = final_line.split(" ")
        if len(words) < 16: raise AssertionError("Make sure log run ran to completion, else discard log as there is no data to collect from.")
        average_precision = float(words[9][:-1])
        roc_auc = float(words[11][:-1])
        final_test_loss = float(words[13][:-1]) 
        accuracy = float(words[15][:-1])
        
    return average_precision, roc_auc, final_test_loss, accuracy, threshold

def plot_thresholds_to_metrics_multiple(date, use_edge_fracs=False):
    average_precisions, roc_aucs, final_test_losses, accuracies, thresholds = collect_metrics_multiple_with_thresholds(date)
    
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    
    for y_values in [average_precisions, roc_aucs, final_test_losses, accuracies]:
        plt.figure()
        # if y_values != final_test_losses: plt.ylim(0, 1)
        # else: plt.ylim(0, 2.5)
        y_str = "Average Precision" if y_values == average_precisions \
            else "ROC AUC" if y_values == roc_aucs \
            else "Final Test Loss" if y_values == final_test_losses \
            else "Accuracy"
        x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
        plt.title(f"{y_str} vs. {x_str}")
        for x_value, y_value in zip(x_values, y_values):
            plt.scatter(x_value, y_value)
        hop_len = len(y_values) // 4
        for i in range(len(y_values) // 4):
            plt.plot([x_values[i], x_values[i + hop_len], x_values[i + 2 * hop_len], x_values[i + 3 * hop_len]], 
                    [y_values[i], y_values[i + hop_len], y_values[i + 2 * hop_len], y_values[i + 3 * hop_len]])


        plt.xlabel(x_str[ : -1]) # remove s character
        plt.ylabel(y_str)
        plt.show()

def plot_thresholds_to_metrics_single_threshold(date, use_edge_fracs=False):
    average_precisions, roc_aucs, final_test_losses, accuracies, thresholds = collect_metrics_multiple_with_thresholds(date)
    
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    
    for y_values in [average_precisions, roc_aucs, final_test_losses, accuracies]:
        plt.figure()
        # if y_values != final_test_losses: plt.ylim(0, 1)
        # else: plt.ylim(0, 2.5)
        y_str = "Average Precision" if y_values == average_precisions \
            else "ROC AUC" if y_values == roc_aucs \
            else "Final Test Loss" if y_values == final_test_losses \
            else "Accuracy"
        x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
        plt.title(f"{y_str} vs. {x_str}")
        for x_value, y_value in zip(x_values, y_values):
            plt.scatter(x_value, y_value)
        plt.xlabel(x_str[ : -1]) # remove s character
        plt.ylabel(y_str)
        plt.show()

def plot_thresholds_to_metrics_single_threshold_box_plot(date, use_edge_fracs=False):
    average_precisions, roc_aucs, final_test_losses, accuracies, thresholds = collect_metrics_multiple_with_thresholds(date)
    
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    
    for y_values in [average_precisions, roc_aucs, final_test_losses, accuracies]:
        plt.figure()
        y_str = "Average Precision" if y_values == average_precisions \
            else "ROC AUC" if y_values == roc_aucs \
            else "Final Test Loss" if y_values == final_test_losses \
            else "Accuracy"
        x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
        plt.title(f"{y_str} vs. {x_str}")
        # hop_len = len(y_values) // 4
        if use_edge_fracs:
            bp = plt.boxplot(y_values, notch=True, patch_artist=True)
        NUM_DIFF_THRESHOLDS = 4
        rgb_colors = [mcolors.hex2color(COLORS[i]) for i in range(NUM_DIFF_THRESHOLDS)]

        if not use_edge_fracs: rgb_colors = rgb_colors[::-1]
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)
        # if use_edge_fracs:
        #     plt.xticks([1, 2, 3, 4], [0.05, 0.06, 0.07, 0.08])
        # else:
        #     plt.xticks([1, 2, 3, 4], [FIVE_PERCENT_THRESHOLD, SIX_PERCENT_THRESHOLD, SEVEN_PERCENT_THRESHOLD, EIGHT_PERCENT_THRESHOLD][::-1])
        plt.xticks([1], [FIVE_PERCENT_THRESHOLD])
        plt.xlabel(x_str[ : -1]) # remove s character
        plt.ylabel(y_str)
        plt.show()


def plot_thresholds_to_metrics_multiple_box_plots(date, use_edge_fracs=False):
    average_precisions, roc_aucs, final_test_losses, accuracies, thresholds = collect_metrics_multiple_with_thresholds(date)
    
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    
    for y_values in [average_precisions, roc_aucs, final_test_losses, accuracies]:
        plt.figure()
        y_str = "Average Precision" if y_values == average_precisions \
            else "ROC AUC" if y_values == roc_aucs \
            else "Final Test Loss" if y_values == final_test_losses \
            else "Accuracy"
        x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
        plt.title(f"{y_str} vs. {x_str}")
        hop_len = len(y_values) // 4
        if use_edge_fracs:
            bp = plt.boxplot([
                [y_values[i + 3 * hop_len] for i in range(len(y_values) // 4)],
                [y_values[i + 2 * hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i + hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i] for i in range(len(y_values) // 4)], 
                ], 
                notch=True,
                patch_artist=True)
        else:
            bp = plt.boxplot([
                [y_values[i] for i in range(len(y_values) // 4)], 
                [y_values[i + hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i + 2 * hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i + 3 * hop_len] for i in range(len(y_values) // 4)],
                ], 
                notch=True,
                patch_artist=True)
        NUM_DIFF_THRESHOLDS = 4
        rgb_colors = [mcolors.hex2color(COLORS[i]) for i in range(NUM_DIFF_THRESHOLDS)]

        if not use_edge_fracs: rgb_colors = rgb_colors[::-1]
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)
        if use_edge_fracs:
            plt.xticks([1, 2, 3, 4], [0.05, 0.06, 0.07, 0.08])
        else:
            plt.xticks([1, 2, 3, 4], [FIVE_PERCENT_THRESHOLD, SIX_PERCENT_THRESHOLD, SEVEN_PERCENT_THRESHOLD, EIGHT_PERCENT_THRESHOLD][::-1])
        plt.xlabel(x_str[ : -1]) # remove s character
        plt.ylabel(y_str)
        plt.show()

def get_age_prediction_metrics_with_thresholds(date, model_type='linear'):
    means_squared_errors, correlations, thresholds = [], [], []
    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    for log_num in os.listdir(date_dir):
        # if "threshold_0.3059" not in log_num: continue 
        # if "threshold_0.3208" not in log_num: continue 
        # if "threshold_0.3384" not in log_num: continue 
        # if "threshold_0.3598" not in log_num: continue 
        # if "second" in log_num or "third" in log_num: continue
        log_file_dir = os.path.join(date_dir, log_num, "log.txt")
        with open(log_file_dir) as log_file:
            log_file_lines = log_file.readlines()
            for log_file_line in log_file_lines:
                if "INFO:root:Using PLV Threshold" in log_file_line:
                    # Index up to -1 to eliminate \n character
                    threshold = float(log_file_line.split(" ")[-1][ : -1])
                    break
            linear_mse_line = log_file_lines[-4]
            linear_correlation_line = log_file_lines[-3]
            ridge_mse_line = log_file_lines[-2]
            ridge_correlation_line = log_file_lines[-1]
            # Index up to -1 to eliminate \n character
            linear_mse = float(linear_mse_line.split(" ")[6][:-1])
            linear_correlation = float(linear_correlation_line.split(" ")[6][ : -1])
            ridge_mse = float(ridge_mse_line.split(" ")[6][:-1])
            ridge_correlation = float(ridge_correlation_line.split(" ")[6][ : -1])

            if model_type == "linear": means_squared_errors.append(linear_mse)
            elif model_type == "ridge": means_squared_errors.append(ridge_mse)
            else: raise ValueError("Invalid model type")
            if model_type == "linear": correlations.append(linear_correlation)
            elif model_type == "ridge": correlations.append(ridge_correlation)
            else: raise ValueError("Invalid model type")
            thresholds.append(threshold)

    return means_squared_errors, correlations, thresholds
