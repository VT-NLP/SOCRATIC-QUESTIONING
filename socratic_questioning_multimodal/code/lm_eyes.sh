clear
python lm_eyes.py \
--data='../data/example_data.jsonl' \
--dataset=aokvqa \
--gpu_id=4 \
--max_depth=2 \
--max_turn_num=2 \
--question_num=3 \
--llm=gpt \
--api="your openai api key"