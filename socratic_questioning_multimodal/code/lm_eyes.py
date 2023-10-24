import os
import json
import torch
import pdb
import time

from thought_tree import ThoughtTree
from thought_node import ThoughtNode
from prompt_map import PromptMap

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--max_depth', type=int)
parser.add_argument('--max_turn_num', type=int)
parser.add_argument('--question_num', type=int)
parser.add_argument('--llm', type=str, default='gpt')
parser.add_argument('--api', type=str, default="")

args = parser.parse_args()

'''Environment Parameter Setup'''

api = args.api # jingyuan
gpu_id = args.gpu_id
os.environ['CUDA_VISIBLE_DEVICES']= str(gpu_id)# set cuda device

'''Framework Hyperparameters'''
# ['vqa','okvqa','aokvqa', 'scienceqa', 'hatefulmeme']
dataset = args.dataset
max_depth = args.max_depth
max_turn_num = args.max_turn_num
num_deeper_question = args.question_num

'''load data'''
def get_save_name(dataset,num_deeper_question,max_depth,max_turn_num):
    name = dataset
    name += '_question' + str(num_deeper_question)
    name += '_depth' + str(max_depth)
    name += '_turn' + str(max_turn_num)
    name += '_' + args.llm
    
    return name

f = open(args.data, 'r')
lines = f.readlines()
f.close()

savedir_name = get_save_name(dataset,num_deeper_question,max_depth,max_turn_num)
print('save_dir_name: ', savedir_name)
if not os.path.exists('../result/' + savedir_name):
    os.mkdir('../result/' + savedir_name)
if not os.path.exists('../result/' + savedir_name + '/inference_log'):
    os.mkdir('../result/' + savedir_name + '/inference_log')
save_dir = '../result/'+ savedir_name


'''initial tree'''
# init prompts
prompt_map_obj = PromptMap()
prompt_map = prompt_map_obj.get_map(dataset)
# init tree
tree = ThoughtTree(gpu_id, api, prompt_map, num_deeper_question,args.llm)
# pdb.set_trace()

'''run'''
summary = {}
print('Save_dir_name: ', savedir_name)
for idx, i in enumerate(lines):
    data = json.loads(i)
    question_id = data['unique_id'].split('_')[-1]
    question = data['question']
    choices = ''
    target = data['target_txt']
    if 'choice' in data:
        choices = ' Choices: ' + str(data['choice'])
        if 'target_id' in data:
            target = data['target_id']
        question = question + choices
    object_regions = None
    if 'object_regions' in data:
        object_regions = data['object_regions']
    img_path = data['image_path']
    # pdb.set_trace()
    
    while True:
        try:
            answer, num_children = tree.run_root_node(save_dir, question_id, question, img_path, max_depth, max_turn_num, object_regions=object_regions)
            break
        # catch error and print it
        except Exception as e:
            print('Error: ',e,'; Restarting... At time: ' + str(time.time()))
            time.sleep(10)
    # pdb.set_trace()
            
    context = tree.root_node.get_context()
    hints = tree.root_node.hint
    data['context'] = context
    data['hint'] = hints
    data['answer'] = answer
    data['num_nodes'] = num_children
    summary[question_id] = data
    with open(save_dir + '/result_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    # print progress
    print('Progress: %d/%d, %d%%' % (idx+1, len(lines), (idx+1)/len(lines)*100), end='\r')
    # break

# pdb.set_trace()



