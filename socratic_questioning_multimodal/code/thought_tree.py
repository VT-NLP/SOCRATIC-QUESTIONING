from thought_node import ThoughtNode
from denseImageCaption import DenseImageCaption
# from vicuna import Vicuna

import os
import json
import pdb


class ThoughtTree:
    def __init__(self, gpu_id, api, prompt_map, num_deeper_question=1,llm='gpt'):        
        # set cuda device
        os.environ['CUDA_VISIBLE_DEVICES']= str(gpu_id)
        # set api
        self.open_api = None
        self.llm = None
        if llm == 'gpt':
            self.open_api = api
        else:
            self.llm = Vicuna()        
         
        self.tools = DenseImageCaption(api,gpu_id,llm=self.llm) # visual aware tools       
        # pdb.set_trace()
        
        self.prompts_map = prompt_map
        self.num_deeper_question = num_deeper_question
        self.root_node = None
        self.root_id = None
        self.save_dir = None
        
    def init_node(self, id, question, if_fact_question, img_path, depth, turn_num, max_depth=2, max_turn_num=2,object_regions=None):
        node = ThoughtNode(self.open_api, id, question, if_fact_question, img_path, depth, turn_num, max_depth, max_turn_num, object_regions=object_regions, llm=self.llm)
        return  node
        
    def init_root_node(self, question, img_path, max_depth=2, max_turn_num=2, object_regions=None):
        root_node = self.init_node(0, question, False, img_path, 1, 1, max_depth, max_turn_num, object_regions=object_regions)
        self.root_node = root_node
    
    def set_context(self, node):
        question = node.question
        img_path = node.img_path
        object_regions = node.object_regions
        description = self.tools.get_visual_descrip(question, img_path)
        node.set_context(description)
      
    def answer_fact_question(self, node):
        answer, continue_deeper = node.answer_fact_question()
        return answer, continue_deeper
      
    def answer_visual_question(self, node):
        answer, continue_deeper = node.answer_visual_question(self.prompts_map)
        return answer, continue_deeper
    
    def run(self, node):
        '''1. set context'''
        if not node.hasContext() and not node.if_fact_question:
            self.set_context(node)
        # return node.get_context()
        '''2. ask question'''
        if node.if_fact_question:
            answer, continue_deeper = self.answer_fact_question(node)
        else:
            answer, continue_deeper = self.answer_visual_question(node)
        # logger
        node_type = 'fact' if node.if_fact_question else 'visual'
        self.log('answer', node, node_type, answer)
        '''3. if continue deeper, then create child node'''
        if continue_deeper:
            # # raise deeper questions    
            num_children = 0
            question_id = 0
            
            for question_type in ['fact', 'visual']:
                additional_infos = []
                # for i in range(self.num_deeper_question):
                deeper_questions = node.raise_question(question_type).split('\n')
                dq_ideas = [dq for dq in deeper_questions if dq[:5] == 'Idea:']
                deeper_questions = [dq for dq in deeper_questions if dq[:5] != 'Idea:']
                for i in range(self.num_deeper_question):
                    if i == len(deeper_questions):
                        break
                    deeper_question = deeper_questions[i]
                    self.log('raise', node, question_type, deeper_question, dq_idea=dq_ideas[i] if i < len(dq_ideas) else None)
                    # create child nodes
                    child = node.create_child_node(question_id, deeper_question, question_type=='fact')
                    question_id += 1
                    # run child nodes
                    additonal_info, num_child = self.run(child)
                    additional_infos.append(additonal_info)
                    num_children += num_child
                    
                # pdb.set_trace()
                for info_idx, info in enumerate(additional_infos):
                    node.set_hint(deeper_questions[info_idx], info, question_type) # add hint from child node
                # pdb.set_trace()
            # run next turn thought
            node.update_turn_num()
            answer, num_child = self.run(node)
            return answer, 1+num_children+num_child
        else:
            return answer, 1
        
    def run_root_node(self, save_dir, root_id, question, img_path, max_depth=2, max_turn_num=2, object_regions=None):
        self.root_id = root_id
        self.save_dir = save_dir
        if os.path.exists(save_dir + '/inference_log/'+self.root_id+'.txt'):
            # remove old log file
            os.remove(save_dir + '/inference_log/'+self.root_id+'.txt')
        self.init_root_node(question, img_path, max_depth=max_depth, max_turn_num=max_turn_num, object_regions=object_regions)
        return self.run(self.root_node)
    
    def log(self, log_type, node, type, response, dq_idea=None):
        depth = node.depth
        turn = node.turn_num
        question_id = node.question_id
        question = node.question
        context = node.context
        if context is not None:
            context = context.strip()
        hint = node.hint
        with open(self.save_dir + '/inference_log/'+self.root_id+'.txt', 'a') as f:
            if log_type == 'answer':
                f.write('=====================Answer a Question=====================\n')
                f.write('Depth: %s\nTurn: %s\nID: %s\nQuestion: %s\nType: %s\nContext: %s\nHint: %s\nAnswer: %s\n\n' % (depth, turn, question_id, question, type, context, hint, response))
            else:
                f.write('=====================Raise Depper=====================\n')
                f.write('Depth: %s\nTurn: %s\nID: %s\nOriginal Question: %s\nType: %s\nContext: %s\nHint: %s\nRaise Reason: %s\nDeeper Question: %s\n\n' % (depth, turn, question_id, question, type, context, hint, dq_idea, response))