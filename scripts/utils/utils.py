import os
import json
import re
from datetime import datetime

def write_to_json(question_ids, img_paths, gts_q, gts_ans, preds_q, preds_ans, json_file_path):
    # Create a list to hold the data
    data = []
    
    # Iterate through the data and combine all elements into a dictionary
    for i in range(len(question_ids)):
        entry = {
            "question_id": question_ids[i],
            "image_path": img_paths[i],
            "ground_truth_question": gts_q[i],
            "ground_truth_answer": gts_ans[i],
            "predicted_question": preds_q[i],
            "predicted_answer": preds_ans[i]
        }
        data.append(entry)
    
    # Write the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def write_to_json_PI(question_ids, img_paths, gts_q, gts_ans, preds, json_file_path):
    # Create a list to hold the data
    data = []
    
    # Iterate through the data and combine all elements into a dictionary
    for i in range(len(question_ids)):
        entry = {
            "question_id": question_ids[i],
            "image_path": img_paths[i],
            "PA_question": gts_q[i],
            "PA_answer": gts_ans[i],
            "predictions": preds[i]
        }
        data.append(entry)
    
    # Write the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data


def write_to_json_eval(question_ids, img_paths, gts_q, gts_ans, preds, json_file_path):
    # Create a list to hold the data
    data = []
    
    # Iterate through the data and combine all elements into a dictionary
    for i in range(len(question_ids)):
        entry = {
            "question_id": question_ids[i],
            "image_path": img_paths[i],
            "GT_question": gts_q[i],
            "GT_answer": gts_ans[i],
            "predictions": preds[i]
        }
        data.append(entry)
    
    # Write the data to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data


# Function to preprocess text
def preprocess(text):
    if text is None:
        return ""
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\s+', '', text)  # Remove spaces
    text = re.sub(r'\W+', '', text)  # Remove special characters
    return text


def return_filePath_PA(args):
    
    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(os.path.dirname(current_file))
    base_results_path = os.path.join(scripts_dir, "results")
    os.makedirs(base_results_path, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")  # e.g. "2025-08-28"
    dated_results_path = os.path.join(base_results_path, today_str)
    os.makedirs(dated_results_path, exist_ok=True)
    json_filename = f"PA_Qwen2vl7b_{args.dataset}.json"
    json_file_path = os.path.join(dated_results_path, json_filename)

    return json_file_path


def return_filePath_PI(model_name, args):

    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(os.path.dirname(current_file))
    base_results_path = os.path.join(scripts_dir, "results")
    os.makedirs(base_results_path, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")  # e.g. "2025-08-28"
    dated_results_path = os.path.join(base_results_path, today_str)
    os.makedirs(dated_results_path, exist_ok=True)
    json_filename = f"PI_Qwen2vl7b_{model_name}_{args.dataset}.json"
    json_file_path = os.path.join(dated_results_path, json_filename)

    return json_file_path


def return_filePath_PI_filter(args):

    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(os.path.dirname(current_file))
    base_results_path = os.path.join(scripts_dir, "results")
    os.makedirs(base_results_path, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")  # e.g. "2025-08-28"
    dated_results_path = os.path.join(base_results_path, today_str)
    os.makedirs(dated_results_path, exist_ok=True)
    json_filename = f"PI_Qwen2vl7b_train_{args.model_path_lvlm.split("/")[1]}_{args.model_path_svlm.split("/")[1]}_{args.dataset}.json"
    json_file_path = os.path.join(dated_results_path, json_filename)

    return json_file_path


def return_filePath_infer(args):

    current_file = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(os.path.dirname(current_file))
    base_results_path = os.path.join(scripts_dir, "results")
    os.makedirs(base_results_path, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")  # e.g. "2025-08-28"
    dated_results_path = os.path.join(base_results_path, today_str)
    os.makedirs(dated_results_path, exist_ok=True)
    json_filename = f"Eval_{args.model_path.split("/")[1]}_{args.dataset}.json"
    json_file_path = os.path.join(dated_results_path, json_filename)

    return json_file_path


def calculate_vqa_accuracy(data):
    """
    Calculate VQA accuracy given a list of data entries.
    
    Args:
        data (list): List of dictionaries with keys:
                     - question id
                     - image path
                     - question
                     - answer
                     - prediction

    Returns:
        float: Accuracy percentage.
    """
    correct = 0
    total = 0

    for entry in data:
        question_id = entry["question_id"]
        gt_answer = preprocess(entry["GT_answer"])
        prediction = preprocess(entry["predictions"])
        print(f"[DEBUG] QID: {question_id} | GT: {gt_answer} | Pred: {prediction}")
        
        if gt_answer in prediction:
            correct += 1
        else:
            pass
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

