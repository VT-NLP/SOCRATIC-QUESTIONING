clear
python socratic_questioning.py \
    --data="../data/example_data.csv" \
    --question_type="lqa" \
    --prompts="../data/prompt_map.json" \
    --question_num=5 \
    --max_turn=3 \
    --max_depth=3 \
    --backbone="gpt" \
    --api="your openai api key" \
    --save_dir=../results 