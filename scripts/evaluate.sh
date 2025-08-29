export CUDA_VISIBLE_DEVICES=1

# python3 utils/evaluate.py --model-path=Qwen/Qwen2-VL-2B-Instruct --dataset=textVQA --batch-size=8   #for textVQA

# python3 utils/evaluate.py --model-path=Qwen/Qwen2-VL-2B-Instruct --dataset=stVQA --batch-size=8   #for stVQA
# python3 utils/evaluate.py --model-path=Qwen/Qwen2-VL-2B-Instruct --dataset=chartVQA --batch-size=8   #for chartVQA
python3 utils/evaluate.py --model-path=Qwen/Qwen2-VL-2B-Instruct --dataset=okVQA --batch-size=8   #for okVQA
