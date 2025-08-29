#required dependencies
import os
import re
import json
import argparse
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.dataLoader import TextVQADataset, STVQADataset, ChartVQADataset, OKVQADataset, collate_fn
from utils.utils import write_to_json, return_filePath_PA

# Check for GPU availability
if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available.")


def PseudoAnnotator(model, processor, data_loader, args):
    
    model.eval()
    preds_q = []
    preds_ans = []
    gts_q = []
    img_paths = []
    gts_ans = []
    question_ids = []
    
    for idx, item in enumerate(data_loader):
        messages = []

        for index in range(len(item["image"])):
            image_path = item["image"][index]  
            question = item["question"][index]
            answer = item["answer"][index]  
            question_id = item["question_id"][index].item()
            
            print("[Ground Truth] |", question_id, "|", image_path, "|", question, "|", answer)
        
            #[TextVQA Prompt]
            if args.dataset == "textVQA" or args.dataset == "stVQA":
                text_prompt = f"""
                The objective is to generate a question-answer pair for a Textual Visual Question Answering (Text-VQA) task. Your
                task is to create a contextually relevant question that directly relate to the image's content, incorporating reasoning or
                direct references to the text, and it's correct answer.

                Output:
                - Question: A natural language question grounded in the image's content and text.
                - Answer: A concise response (single word, phrase, or Yes/No) derived from the text or reasoning based on it.
                """
            elif args.dataset == "chartVQA":
                text_prompt = f"""
                The objective is to generate a question-answer pair for a Chart Visual Question Answering (ChartVQA) task.
                Your task is to create a contextually relevant question that directly relates to the content of a given chart, incorporating
                reasoning based on the visualized data.

                Output Requirements:
                - Question: A natural language question grounded in the chart's content, requiring numerical reasoning, trend analysis,
                or data lookup.
                - Answer: A concise response (single word, number, phrase, or Yes/No) derived from the chart’s data.
                
                Guidelines for Question Generation:
                1. Direct Lookup Questions – Questions that require extracting specific values from the chart.
                2. Comparison Questions – Questions that require comparing values between different categories.
                3. Trend & Pattern Recognition – Questions about trends, increases, decreases, or correlations in the data.
                4. Inference-Based Questions – Questions that require reasoning beyond direct value lookup.
                
                Ensure the question is meaningful and the answer is accurate based on the chart data.
                """
            elif args.dataset == "okVQA":
                text_prompt = f"""
                The objective is to generate a question-answer pair for a Knowledge-based Visual Question Answering (K-VQA) task.
                Your task is to create a contextually relevant question that directly relates to the image's content while requiring external
                world knowledge to answer correctly, and it's correct answer.

                Output Requirements:
                - Question: A natural language question grounded in the image’s content but requiring reasoning beyond direct perception,
                incorporating real-world knowledge.
                - Answer: A single-word response based on general world knowledge.

                Guidelines for Question Generation:
                1. Object & Scene Understanding – Questions that require identifying objects or actions in the image and connecting them
                to broader knowledge.
                2. Commonsense Reasoning – Questions that require logical deductions about the scene.
                3. Cultural & Historical Context – Questions related to well-known historical events, traditions, or cultural references.
                4. Scientific & Factual Knowledge – Questions involving basic physics, biology, geography, or general knowledge.
                5. Everyday Life & Social Understanding – Questions about daily activities, professions, or human behaviors.
                
                Ensure that the generated question is meaningful and requires external knowledge beyond just the image’s visual content.
                """
            else:
                print("Invalid Dataset")
                exit(0)

            # Prepare input for Qwen2-VL
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
        
        #print("[BATCH] | [Predicted Truth] |", question_id, "|", image_path, "|", generated_texts)
        
        for index in range(len(generated_texts)):
            
            question_id = item["question_id"][index].item()
            image_path = item["image"][index]
            genText = generated_texts[index]
            question = item["question"][index]
            answer = item["answer"][index]  # Ground truth answers
            
            question_match = re.search(r"Question:\s*(.*?)\?", genText, re.IGNORECASE)
            answer_match = re.search(r"Answer:\s*(.*)", genText, re.IGNORECASE)

            generated_question = question_match.group(1).strip() if question_match else None
            generated_answer = answer_match.group(1).strip() if answer_match else None

            preds_q.append(generated_question)
            preds_ans.append(generated_answer)
            gts_q.append(question)
            img_paths.append(image_path)
            gts_ans.append(answer)
            question_ids.append(question_id)

            print(f"[Predicted Truth] | {question_id} | {image_path} | {generated_question} | {generated_answer}")
                
        torch.cuda.empty_cache()  # Clear memory after each iteration
        break
    
    json_file_path = return_filePath_PA(args)
    write_to_json(question_ids, img_paths, gts_q, gts_ans, preds_q, preds_ans, json_file_path)
    print(f"Results saved to {json_file_path}")


def main(args):

    if args.dataset == "textVQA":
        # Initialize the dataset and dataloader
        dataset = TextVQADataset(json_path="/workspace/scripts/27_08_2025_code_release/data/TextVQA/TextVQA_0.5.1_unique_train.json", image_dir="/workspace/data/TextVQA/train_images")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    elif args.dataset == "stVQA":
        # Initialize dataset
        dataset = STVQADataset(json_path='/workspace/scripts/27_08_2025_code_release/data/STVQA/train_task_1_onePerImage_train.json', image_dir='/workspace/data/ST-VQA/train/data')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    elif args.dataset == "chartVQA":
        # Initialize dataset
        dataset = ChartVQADataset(json_path='/workspace/scripts/27_08_2025_code_release/data/ChartVQA/train_onePerImage.json', image_dir='/workspace/data/ChartVQA/ChartQA_Dataset/train-val')
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn) 
    
    elif args.dataset == "okVQA":
        # Initialize dataset
        dataset = OKVQADataset(json_path='/workspace/scripts/27_08_2025_code_release/data/OKVQA/okvqa_train_unique_combine.json', image_dir="/workspace/data/okvqa/images/train2014")
        data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    else:
        print("Invalid Dataset")
        exit(0)
    

    # Load Qwen2-VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=((128 * 28 * 28)), max_pixels=((512 * 28 * 28)))

    # Generate pseudo annotations 
    PseudoAnnotator(model, processor, data_loader, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the dataset')

    args = parser.parse_args()
    main(args)

