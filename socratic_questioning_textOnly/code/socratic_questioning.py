import pandas as pd
import numpy as np
import json
import re
import time
import os
import pdb
import ast
import argparse

from socratic_tree import SocraticTree

# functions to load csv data
def csv_to_list(csv_path):
    # load csv, no header
    df = pd.read_csv(csv_path, header=None)
    # add header, [question, option1, option2, option3, option4, answer]
    if len(df.columns) == 6: # mmmu
        df.columns = ['question', 'option1', 'option2', 'option3', 'option4', 'answer']
        data_dict = [{
            'question': df['question'][i],
            'options': [df['option1'][i], df['option2'][i], df['option3'][i], df['option4'][i]],
            'target': df['answer'][i]
        } for i in range(len(df))]
    elif len(df.columns) == 8: # lqa
        df.columns = ['id', 'context', 'question', 'option1', 'option2', 'option3', 'option4', 'answer']
        data_dict = [{
            'context': df['context'][i],
            'question': df['question'][i],
            'options': [df['option1'][i], df['option2'][i], df['option3'][i], df['option4'][i]],
            'target': df['answer'][i]
        } for i in range(len(df))]
    elif len(df.columns) == 7: # cqa
        df.columns = ['question', 'option1', 'option2', 'option3', 'option4', 'option5', 'answer']
        data_dict = [{
            'question': df['question'][i],
            'options': [df['option1'][i], df['option2'][i], df['option3'][i], df['option4'][i], df['option5'][i]],
            'target': df['answer'][i]
        } for i in range(len(df))]
    elif len(df.columns) == 2: # prm800k
        df.columns = ['question', 'answer']
        data_dict = [{
            'question': df['question'][i],
            'target': df['answer'][i]
        } for i in range(len(df))]
        
    return data_dict

def add_optionID_toList(options):
    option_ls = []
    for i, option in enumerate(options):
        if i==0:
            label = 'A'
        elif i==1:
            label = 'B'
        elif i==2:
            label = 'C'
        elif i==3:
            label = 'D'
        elif i==4:
            label = 'E'
        option_ls.append(label + '. ' + option)
    return option_ls

# main
args = argparse.ArgumentParser()
args.add_argument('--data', type=str)
args.add_argument('--question_type', type=str)
args.add_argument('--prompts', type=str)
args.add_argument('--question_num', type=int, default=5)
args.add_argument('--max_turn', type=int, default=3)
args.add_argument('--max_depth', type=int, default=3)
args.add_argument('--backbone', type=str, default="gpt")
args.add_argument('--api', type=str, default="")
args.add_argument('--save_dir', type=str, default="../result")
args = args.parse_args()

# load data
data = csv_to_list(args.data)
prompt_map = json.load(open(args.prompts, 'r'))
hyperparameter = [args.question_num, args.max_turn, args.max_depth]
backbone = args.backbone
openai_api = args.api

num_question = hyperparameter[0]
max_turn = hyperparameter[1]
max_depth = hyperparameter[2]
hyperparameter_str = '_q' + str(num_question) + '_t' + str(max_turn) + '_d' + str(max_depth)
data_name = os.path.basename(args.data).split('.')[0]
save_dir = args.save_dir+'/'+data_name+hyperparameter_str
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir+'/log'):
    os.makedirs(save_dir+'/log')

print('\nLoad data from: ', args.data)
print('Question type: ', args.question_type)
print('Load prompts from: ', args.prompts)
print('Deep question number: ', num_question)
print('Max turn: ', max_turn)
print('Max depth: ', max_depth)
print('Backbone: ', backbone)
print('Save results to: ', save_dir, '\n')

# init tree
socatic_tree = SocraticTree(backbone, openai_api, prompt_map, args.question_type, num_question, max_turn, max_depth, save_dir)
# pdb.set_trace()

# iterate all questions
t0 = time.time()
result = []
for i, dp in enumerate(data):
    dp['id'] = i
    context = None
    if 'context' in dp:
        context = dp['context']
    question = dp['question']
    if 'options' in dp:
        options =  dp['options']
    else:
        options = None
    
    # answer question (answer option ID)
    answer, node, hints = socatic_tree.start(i, question, options, context=context)

    # confidence, raw_confidence = check_confidence(question, options, hints, raw_answer, system_define)
    if options is not None:
        dp['options'] = add_optionID_toList(options)
    dp['prediction'] = answer
    dp['hints'] = hints
    dp['grade'] = 1 if answer == dp['target'] else 0
    # sort keys in dp with order: id, question, options, target, prediction, hints, grade
    if options is not None:
        if 'context' in dp:
            dp = {key: dp[key] for key in ['id', 'context', 'question', 'options', 'target', 'prediction', 'hints', 'grade']}
        else:
            dp = {key: dp[key] for key in ['id', 'question', 'options', 'target', 'prediction', 'hints', 'grade']}
    else:
        if 'context' in dp:
            dp = {key: dp[key] for key in ['id', 'context', 'question', 'target', 'prediction', 'hints', 'grade']}
        else:
            dp = {key: dp[key] for key in ['id', 'question', 'target', 'prediction', 'hints', 'grade']}
    result.append(dp)
    
    # save result
    with open(save_dir + '/result_summary.json', 'w') as f:
        json.dump(result, f, indent=4)
    
    # print progress in percentage, end with \r to overwrite
    print('Dataset: '+ args.question_type, 'Progress: ' + str(round((i+1)/len(data)*100, 2)) + '%', end='\r')
    # pdb.set_trace()
    # break
    
# save result
with open(save_dir + '/result_summary.json', 'w') as f:
    json.dump(result, f, indent=4)
    
t1 = time.time()
print('Time: ' + str(t1-t0) + ' s')

