# SOCRATIC-QUESTIONING
Implementation for the paper: [The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models](https://arxiv.org/abs/2305.14999). This paper has been accepted by EMNLP 2023.

## Install
```sh
pip install openai
```
For the multimodal implementation, please follow the official instructions to install [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

## Run
Text-Only:
1. Prepare the data. Input data should be given as a CSV file. Question ID, Context, Question, Option A, Option B, Option C, Option D, and Ground Truth are split by comma. An example input file is provided: ```./socratic_questioning_textOnly/data/example_data.csv```
2. Add your own prompts into the prompts file. Prompts for tasks and datasets presented in the paper have already been written. 
```./socratic_questioning_textOnly/data/prompt_map.json```
3. Replace the argument of question_type with your dataset name, and replace the OpenAI API with your own.
```./socratic_questioning_textOnly/code/socratic_questioning.sh```
4. Run ```sh ./socratic_questioning_textOnly/code/socratic_questioning.sh```. The output file will be in ```./socratic_questioning_textOnly/results```.

MultiModal:
1. Prepare the data. Input data should be given as a JSONL file. The default image store path is: `./socratic_questioning_multimodal/data/imgs`. An example input file is provided: ```./socratic_questioning_multimodal/data/example_data.jsonl```
2. Replace the argument of dataset with your dataset name, and replace the OpenAI API with your own.
```./socratic_questioning_multimodal/code/lm_eyes.sh```
4. Run ```sh ./socratic_questioning_multimodal/code/lm_eyes.sh```. The output file will be in ```./socratic_questioning_multimodal/result```.