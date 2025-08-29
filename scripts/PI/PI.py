#required dependencies
import os
import json 
import argparse
from datetime import datetime
import csv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader

from utils.dataLoader import TextVQADataset_selfEval, STVQADataset_selfEval, ChartVQADataset_selfEval, OKVQADataset_selfEval, collate_fn
from utils.utils import write_to_json_PI, preprocess, return_filePath_PI, return_filePath_PI_filter

# Check for GPU availability
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")


def write_to_csv(question_ids, img_paths, questions, answers, predictions, csv_file_path):
    """
    Writes the given data (image paths, questions, answers, and predictions) to a CSV file.

    Args:
    - img_paths (list): List of image paths.
    - questions (list): List of questions.
    - answers (list): List of answers.
    - predictions (list): List of predictions.
    - csv_file_path (str): The path to the CSV file where data will be saved.

    Returns:
    - None
    """
    # Open the CSV file in write mode
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Writing the header row
        writer.writerow(["Question_ID", "Image Path", "Question", "Answer", "Prediction"])
        
        # Writing each row of data
        for question_id, img_path, question, answer, prediction in zip(question_ids, img_paths, questions, answers, predictions):
            writer.writerow([question_id, img_path, question, answer, prediction])

    print(f"Data has been written to {csv_file_path}")


def Infer(model_name, model, processor, data_loader, args):
    
    model.eval()
    preds = []
    questions = []
    img_paths = []
    answers = []
    question_ids = []
        
    for idx, item in enumerate(data_loader):
        messages = []

        for index in range(len(item["image"])):
            question_id = item["question_id"][index].item()
            image_path = item["image"][index]
            question = item["question"][index]
            answer = item["answer"][index]  
            
            print("[Ground Truth] |", question_id, "|", image_path, "|", question, "|", answer)

            text_prompt = question + "Answer the following question in a single word or phrase."
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            messages.append(message)
                    
        # Prepare inputs for the Qwen2-VL model
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
         
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
            
        with torch.no_grad():
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        for index in range(len(generated_texts)):

            question_id = item["question_id"][index].item()
            image_path = item["image"][index]
            question = item["question"][index]
            answer = item["answer"][index]
            generated_text = generated_texts[index]

            preds.append(generated_text)
            questions.append(question)
            img_paths.append(image_path)
            answers.append(answer)
            question_ids.append(question_id)

            print("[Predicted Truth] |", question_id, image_path, question, answer, generated_text)
                    
        torch.cuda.empty_cache()  # Clear memory after each iteration
        break

    json_file_path = return_filePath_PI(model_name, args)
    output = write_to_json_PI(question_ids, img_paths, questions, answers, preds, json_file_path)
    print(f"Results saved to {json_file_path}")

    return output


def Parity_Identifier(output_lvlm, output_svlm, args):
    svlm_dict = {str(entry["question_id"]): entry for entry in output_svlm}
    lvlm_dict = {str(entry["question_id"]): entry for entry in output_lvlm}

    svlm_keys = set(svlm_dict.keys())
    lvlm_keys = set(lvlm_dict.keys())

    only_in_svlm = svlm_keys - lvlm_keys
    only_in_lvlm = lvlm_keys - svlm_keys
    common_keys = svlm_keys & lvlm_keys  # Intersection gives common keys

    # if only_in_svlm or only_in_lvlm:
    #     print(f"Keys in svlm but not in lvlm ({len(only_in_svlm)}): {sorted(only_in_svlm)[:10]}")
    #     print(f"Keys in lvlm but not in svlm ({len(only_in_lvlm)}): {sorted(only_in_lvlm)[:10]}")
    #     print(f"Common keys count: {len(common_keys)}")

    filtered_question_ids = []

    # Iterate over the question IDs
    for qid in svlm_dict:
        if qid in svlm_dict and qid in lvlm_dict:
            svlm_entry = svlm_dict[qid]
            lvlm_entry = lvlm_dict[qid]
            
            # print(q2b_entry, q7b_entry)

            gt_answer = preprocess(svlm_entry["PA_answer"])  # GT answer from the JSON
            pred_svlm = preprocess(svlm_entry.get("predictions", None))
            pred_lvlm = preprocess(lvlm_entry.get("predictions", None))
            print("DEBUG", gt_answer, pred_svlm, pred_lvlm)

            # Check if svlm failed but lvlm succeeded
            if gt_answer not in pred_svlm and gt_answer in pred_lvlm:
                entry = {
                    "id": svlm_entry["question_id"],
                    "image": svlm_entry["image_path"],
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{svlm_entry['PA_question']}"
                        },
                        {
                            "from": "gpt",
                            "value": svlm_entry["PA_answer"]
                        }
                    ]
                }
                filtered_question_ids.append(entry)

    json_file_path = return_filePath_PI_filter(args)

    with open(json_file_path, 'w') as f:
        json.dump(filtered_question_ids, f, indent=2)
    print(f"Results saved to {json_file_path}")

    return None


def main(args):

    if args.dataset == "textVQA":
        # Initialize the dataset and dataloader
        dataset = TextVQADataset_selfEval(json_path="/workspace/scripts/27_08_2025_code_release/data/TextVQA/PA_backup/Qwen2vl7b/output.json")  #note: here put the path of json file created during the PA step
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    elif args.dataset == "stVQA":
        # Initialize dataset
        dataset = STVQADataset_selfEval(json_path='/workspace/scripts/27_08_2025_code_release/data/STVQA/PA_backup/Qwen2vl7b/output.json') #note: here put the path of json file created during the PA step
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    elif args.dataset == "chartVQA":
        # Initialize dataset
        dataset = ChartVQADataset_selfEval(json_path='/workspace/scripts/27_08_2025_code_release/data/ChartVQA/PA_backup/Qwen2vl7b/output.json') #note: here put the path of json file created during the PA step
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    elif args.dataset == "okVQA":
        # Initialize dataset
        dataset = OKVQADataset_selfEval(json_path='/workspace/scripts/27_08_2025_code_release/data/OKVQA/PA_backup/Qwen2vl7b/output.json') #note: here put the path of json file created during the PA step
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    else:
        print("Invalid Dataset")
        exit(0)
    

    # Load Qwen2-VL model and processor
    model_lvlm = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path_lvlm, torch_dtype="auto", device_map="auto")
    processor_lvlm = AutoProcessor.from_pretrained(args.model_path_lvlm, min_pixels=((128 * 28 * 28)), max_pixels=((512 * 28 * 28)))

    model_svlm = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path_svlm, torch_dtype="auto", device_map="auto")
    processor_svlm = AutoProcessor.from_pretrained(args.model_path_svlm, min_pixels=((128 * 28 * 28)), max_pixels=((512 * 28 * 28)))

    output_lvlm = Infer(args.model_path_lvlm.split("/")[1], model_lvlm, processor_lvlm, data_loader, args)
    output_svlm = Infer(args.model_path_svlm.split("/")[1], model_svlm, processor_svlm, data_loader, args)
    Parity_Identifier(output_lvlm, output_svlm, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters
    parser.add_argument('--model-path-lvlm', type=str, default='Qwen/Qwen2-VL-7B-Instruct',
                        help='Path for lvlm model.')
    parser.add_argument('--model-path-svlm', type=str, default='Qwen/Qwen2-VL-2B-Instruct',
                        help='Path for svlm model.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset')

    args = parser.parse_args()
    main(args)
