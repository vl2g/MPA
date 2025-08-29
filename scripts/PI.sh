export CUDA_VISIBLE_DEVICES=1

python3 PI/PI.py --model-path-lvlm=Qwen/Qwen2-VL-7B-Instruct --model-path-svlm=Qwen/Qwen2-VL-2B-Instruct  --dataset=textVQA --batch-size=4   #for textVQA

# python3 PI/PI.py --model-path-lvlm=Qwen/Qwen2-VL-7B-Instruct --model-path-svlm=Qwen/Qwen2-VL-2B-Instruct  --dataset=stVQA --batch-size=4   #for stVQA
# python3 PI/PI.py --model-path-lvlm=Qwen/Qwen2-VL-7B-Instruct --model-path-svlm=Qwen/Qwen2-VL-2B-Instruct  --dataset=chartVQA --batch-size=4   #for chartVQA
# python3 PI/PI.py --model-path-lvlm=Qwen/Qwen2-VL-7B-Instruct --model-path-svlm=Qwen/Qwen2-VL-2B-Instruct  --dataset=okVQA --batch-size=4   #for okVQA