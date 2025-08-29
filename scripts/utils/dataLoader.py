#required dependencies
import os
import json

from collections import Counter

import torch
from torch.utils.data import Dataset


# Collate functions for DataLoader
def collate_fn(batch):
    
    images = []
    questions = []
    answers = []
    question_ids = []

    for item in batch:
        images.append(item['image'])
        questions.append(item['question'])
        answers.append(item['answer'])
        question_ids.append(item['question_id'])

    return {
        'image': images,                          # Tensor of images
        'question': questions,                    # List of questions (string)
        'answer': answers,                        # List of answers (string)
        'question_id': torch.tensor(question_ids) # Tensor of question IDs
    }


# Dataset class for TextVQA
class TextVQADataset(Dataset):
    def __init__(self, json_path, image_dir):
        """
        Args:
            json_path (str): Path to the TextVQA train split(unique) JSON file.
            image_dir (str): Directory containing the images.
        """

        with open(json_path, 'r') as f:
            self.data = json.load(f)["data"]  # Load the list of QA data

        self.image_dir = image_dir


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct image path from the image_id
        image_path = os.path.join(self.image_dir, f"{item['image_id']}.jpg")  # Assuming images are in .jpg format
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        question = item['question']
        question_id = item['question_id']
        
        # Find the most frequent answer
        answer = Counter(item['answers']).most_common(1)[0][0]

        return {
            'image': image_path,
            'question': question,
            'answer': answer,
            'question_id': question_id
        }

# Dataset class for ST-VQA
class STVQADataset(Dataset):
    def __init__(self, json_path, image_dir, set_name='train', transform=None):
        """
        Args:
            json_path (str): Path to the ST-VQA train split(unique) JSON file.
            image_root (str): Root directory containing subfolders with images.
        """

        with open(json_path, 'r') as f:
            self.data = json.load(f)["data"]
                
        self.image_dir = image_dir


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        # Construct full image path
        image_path = os.path.join(self.image_dir, item['file_path'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        question = item['question']
        question_id = item['question_id']

        #take the most frequent answer as the answer
        answers = Counter(item['answers']).most_common(1)[0][0]

        return {
            'image': image_path,
            'question': question,
            'answer': answers,
            'question_id': question_id
            }

# Dataset class for Chart-VQA
class ChartVQADataset(Dataset):
    def __init__(self, json_path, image_dir):
        """
        Args:
            json_path (str): Path to the Chart-VQA train split(unique) JSON file.
        """
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)  

        self.image_dir = image_dir


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = os.path.join(self.image_dir, f"{item['imgname']}")  # Assuming images are in .jpg format
        
        question = item['query']
        question_id = item["question_id"]
        answer = item['label']
        
        return {
            'image': image_path,
            'question': question,
            'answer': answer,
            'question_id': question_id
        }

# Dataset class for OK-VQA
class OKVQADataset(Dataset):
    def __init__(self, json_path, image_dir):
        """
        Args:
            json_path (str): Path to the OK-VQA train split(unique) JSON annotation file.
            image_dir (str): Directory containing COCO-style image files.
        """

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.image_dir = image_dir

        self.samples = []
        for entry in raw_data:
            image_id = entry["image_id"]
            for qa in entry["qas"]:
                self.samples.append({
                    "image_id": image_id,
                    "question_id": qa["question_id"],
                    "question": qa["question"],
                    "answers": qa["answers"]
                })


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample["image_id"]

        # Convert image_id to COCO-style file name with zero-padding
        image_filename = f"COCO_train2014_{int(image_id):012d}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)

        answer = Counter(sample['answers']).most_common(1)[0][0]
        
        return {
            "image": image_path,
            "question": sample["question"],
            "answer": answer,
            "question_id": sample["question_id"]
        }

class OKVQADataset_eval(Dataset):
    def __init__(self, json_path, image_dir):
        """
        Args:
            json_path (str): Path to the OK-VQA train split(unique) JSON annotation file.
            image_dir (str): Directory containing COCO-style image files.
        """

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.image_dir = image_dir

        self.samples = []
        for entry in raw_data:
            image_id = entry["image_id"]
            self.samples.append({
                "image_id": image_id,
                "question_id": entry["question_id"],
                "question": entry["question"],
                "answers": entry["answers"]
            })


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample["image_id"]

        # Convert image_id to COCO-style file name with zero-padding
        image_filename = f"COCO_val2014_{int(image_id):012d}.jpg"
        image_path = os.path.join(self.image_dir, image_filename)

        answer = Counter(sample['answers']).most_common(1)[0][0]
        
        return {
            "image": image_path,
            "question": sample["question"],
            "answer": answer,
            "question_id": sample["question_id"]
        }


# Dataset class for TextVQA (for self-evaluation purposes)
class TextVQADataset_selfEval(Dataset):
    def __init__(self, json_path):
        """
        Args:
            json_path (str): Path to the TextVQA PA JSON file.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)  # Load the list of QA data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item['image_path']  
        question = item['predicted_question']
        question_id = int(item['question_id'])
        answer = item['predicted_answer']

        return {
            'image': image_path,
            'question': question,
            'answer': answer,
            'question_id': question_id
        }

# Dataset class for ST-VQA (for self-evaluation purposes)
class STVQADataset_selfEval(Dataset):
    def __init__(self, json_path):
        """
        Args:
            json_path (str): Path to the ST-VQA PA JSON file.
=        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]

        # Construct full image path
        image_path = item['image_path']
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        question = item['predicted_question']
        answers = item['predicted_answer']
        question_id = int(item['question_id'])

        return {
            'image': image_path,
            'question': question,
            'answer': answers,
            'question_id': question_id
            }

# Dataset class for Chart-VQA (for self-evaluation purposes)
class ChartVQADataset_selfEval(Dataset):
    def __init__(self, json_path):
                
        with open(json_path, 'r') as f:
            self.data = json.load(f)  


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item["image_path"]
        question_id = item['question_id']
        question = item['predicted_question']
        answer = item['predicted_answer']
        
        return {
            'question_id': question_id,
            'image': image_path,
            'question': question,
            'answer': answer
        }

# Dataset class for OK-VQA (for evaluation purposes)
class OKVQADataset_selfEval(Dataset):
    def __init__(self, json_path):
                
        with open(json_path, 'r') as f:
            self.data = json.load(f)  


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item["image_path"]
        question_id = item['question_id']
        question = item['predicted_question']
        answer = item['predicted_answer']
        
        return {
            'question_id': question_id,
            'image': image_path,
            'question': question,
            'answer': answer
        }

