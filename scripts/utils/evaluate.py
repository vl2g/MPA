#required dependencies
import os
import json 
import argparse
from datetime import datetime
import csv

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader

from dataLoader import TextVQADataset, STVQADataset, ChartVQADataset, OKVQADataset_eval, collate_fn
from utils import write_to_json_eval, preprocess, return_filePath_infer, calculate_vqa_accuracy

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


def Infer(model, processor, data_loader, args):
    
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

    json_file_path = return_filePath_infer(args)
    output = write_to_json_eval(question_ids, img_paths, questions, answers, preds, json_file_path)
    print(f"Results saved to {json_file_path}")

    acc = calculate_vqa_accuracy(output)
    print(f"VQA Accuracy: {acc:.2f}%")

    return acc


def main(args):

    if args.dataset == "textVQA":
        # Initialize the dataset and dataloader
        dataset = TextVQADataset(json_path="/workspace/scripts/27_08_2025_code_release/data/TextVQA/TextVQA_0.5.1_val.json", image_dir="/workspace/data/TextVQA/train_images")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    elif args.dataset == "stVQA":
        # Initialize dataset
        dataset = STVQADataset(json_path='/workspace/scripts/27_08_2025_code_release/data/STVQA/train_task_1_onePerImage_val.json', image_dir='/workspace/data/ST-VQA/train/data')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    elif args.dataset == "chartVQA":
        # Initialize dataset
        dataset = ChartVQADataset(json_path='/workspace/scripts/27_08_2025_code_release/data/ChartVQA/test_combined.json', image_dir='/workspace/data/ChartVQA/ChartQA_Dataset/test/png')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn) 
    
    elif args.dataset == "okVQA":
        # Initialize dataset
        dataset = OKVQADataset_eval(json_path='/workspace/scripts/27_08_2025_code_release/data/OKVQA/okvqa_val_combine.json', image_dir="/workspace/data/okvqa/images/val2014")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    else:
        print("Invalid Dataset")
        exit(0)
    

    # Load Qwen2-VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=((128 * 28 * 28)), max_pixels=((512 * 28 * 28)))

    acc = Infer(model, processor, data_loader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen2-VL-2B-Instruct',
                        help='Path for model.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset')

    args = parser.parse_args()
    main(args)
