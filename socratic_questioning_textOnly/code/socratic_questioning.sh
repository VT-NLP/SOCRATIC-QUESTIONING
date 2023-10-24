clear
python socratic_questioning.py \
    --data="../data/example_data.csv" \
    --question_type="lqa" \
    --prompts="../data/prompt_map.json" \
    --question_num=5 \
    --max_turn=3 \
    --max_depth=3 \
    --backbone="gpt" \
    --api="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" \
    --save_dir=../results 