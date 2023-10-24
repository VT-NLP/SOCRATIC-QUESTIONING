from socratic_node import SocraticNode

import os
import json
import re
import pdb
import torch
import time
import openai

class SocraticTree:
    def __init__(self, backbone, api, prompt_map, dataset, num_question, max_turn, max_depth, save_dir):
        self.backbone = backbone
        if backbone == 'gpt':
            openai.api_key = api
        elif backbone == 'falcon':
            # host model by text-generation: 
            from text_generation import Client
            self.pipeline = Client("http://127.0.0.1:8080")
            
        self.prompt_map = prompt_map
        self.dataset = dataset
        self.num_question = num_question
        self.max_turn = max_turn
        self.max_depth = max_depth
        
        self.root_node = None
        
        self.save_dir = save_dir    
        self.log_path = None
    
        
    def select_prompt(self, isMultipleChoice, type):
        try:
            # question_type = 'multipleChoice' if isMultipleChoice else 'normalQA'
            question_type = 'multipleChoice_cot' if isMultipleChoice else 'normalQA'
            prompt = self.prompt_map[type][question_type][self.dataset]['prompt']
            tip = self.prompt_map[type][question_type][self.dataset]['tip']
            system_define = self.prompt_map[type][question_type][self.dataset]['system_define']
        except Exception as e:
            print(e)
            pdb.set_trace()
        return prompt, tip, system_define
    
    def request_gpt(self, system_define, prompt):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    # model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_define},
                        {"role": "user", "content": prompt},
                    ]
                )                   
                break
            except Exception as e:
                # current time in human-readable format
                obj = time.localtime()
                t = time.asctime(obj)
                print(t, 'Request failed. Retrying...')
                pdb.set_trace()
                time.sleep(10)
                
        return response['choices'][0]['message']['content'].strip()
    
    def request_falcon(self, system_define, prompt):
        input_str = system_define + '\n\n\n' + prompt
        answer = self.pipeline.generate(input_str, max_new_tokens=300).generated_text 
        return answer.strip()
    
    def request(self, system_define, prompt):
        if self.backbone == 'gpt':
            return self.request_gpt(system_define, prompt)
        elif self.backbone == 'falcon':
            return self.request_falcon(system_define, prompt)
    
    def qa2hint(self, question, answer):
        # request chatGPT
        system_define = """Imagine you are an editor. You are given a question-and-answer pair. You need to merge the question and answer into a statement sentence."""
        prompt = "Question: " + question + "\nAnswer: " + answer
        return self.request(system_define, prompt)
    
    def get_clean_answer(self, raw_answer):
        answer = raw_answer
        # remove " in the raw_answer string
        while '"' in answer:
            answer = answer.replace('"', '')
        if ': ' in answer:
            answer = answer.split(': ')[1]
        if 'the answer is ' in answer:
            idx = answer.index('the answer is ')
            answer = answer[idx+14]
        if 'Option ' in answer:
            answer = answer.split('Option ')[1][0]
        if len(answer) > 0 and answer[-1] == '.':
            answer = answer[:-1]
        if 'A. ' in answer:
            answer = 'A'
        elif 'B. ' in answer:
            answer = 'B'
        elif 'C. ' in answer:
            answer = 'C'
        elif 'D. ' in answer:
            answer = 'D'
        elif '. ' in raw_answer:
            answer = raw_answer.split('. ')[1]
        if len(answer) == 2 and answer[1] == '.':
            answer = answer[0]
        return answer

    def answer_question(self, node):
        # if False:
        if node.depth == 1 and node.hasHint():
            dict_answer_candidates = {}
            for i, hint in enumerate(node.hints):
                format_hint = '(1) ' + hint + '.'   
                # generate prompt
                prompt, tip, system_define = self.select_prompt(node.isMultipleChoice(), 'answer')
                if node.context is not None:
                    prompt += 'Context: ' + node.context + '\n'
                prompt += 'Question: '
                prompt += node.question + '\n'
                if node.isMultipleChoice():
                    prompt += 'Option: ' + node.add_optionID_toText() + '\n'
                if node.hasHint():
                    prompt += 'Hints: ' + format_hint + '\n'
                prompt += tip + '\n'
                prompt += 'Answer: '
                
                # request chatGPT
                raw_answer = self.request(system_define, prompt)
                
                # post-process for socratic cot
                answer = raw_answer
                if ' answer is: ' in raw_answer:
                    # locate position, and the next charactor is the answer
                    idx = raw_answer.index(' answer is: ')
                    answer = raw_answer[idx+12:]
                elif ' answer is ' in raw_answer:
                    idx = raw_answer.index(' answer is ')
                    answer = raw_answer[idx+11:]
                elif ' option is: ' in raw_answer:
                    # locate position, and the next charactor is the answer
                    idx = raw_answer.index(' option is: ')
                    answer = raw_answer[idx+12:]
                elif ' option is ' in raw_answer:
                    idx = raw_answer.index(' option is ')
                    answer = raw_answer[idx+11:]
                answer_parts = answer.split(';')
                
                optionID_part = answer_parts[0].strip() # Answer: A
                if ': ' in optionID_part:
                    answer = optionID_part.split(': ')[1].strip() # A
                else:
                    answer = optionID_part.strip()
                if node.isMultipleChoice():
                    answer = self.get_clean_answer(answer)
                    
                hints_part = None
                if len(answer_parts) > 1:
                    hints_part = answer_parts[1].strip() # Used hints: 1,2,3
                if hints_part is not None:
                    if ': ' in hints_part:
                        used_hints = hints_part.split(': ')[1].strip()
                    else:
                        used_hints = hints_part.strip()
                else:
                    used_hints = None
                    
                confidence_part = None
                if len(answer_parts) > 2:
                    confidence_part = answer_parts[2].strip() 
                if confidence_part is not None:
                    if 'high' in confidence_part or 'High' in confidence_part:
                        confidence = 'high'
                    elif 'middle' in confidence_part or 'Middle' in confidence_part:
                        confidence = 'middle'
                    elif 'low' in confidence_part or 'Low' in confidence_part:
                        confidence = 'low'
                    else:
                        confidence = 'middle'
                else:
                    confidence = 'low'
                    
                if answer not in dict_answer_candidates:
                    dict_answer_candidates[answer] = []                    
                dict_answer_candidates[answer].append([raw_answer, used_hints, confidence])
            
            # get key whose value has the largest length
            answer = max(dict_answer_candidates, key=lambda k: len(dict_answer_candidates[k]))
            raw_answer = dict_answer_candidates[answer][0][0]
            used_hints = dict_answer_candidates[answer][0][1]
            ls_confidence = [item[2] for item in dict_answer_candidates[answer]]
            confidence = max(set(ls_confidence), key = ls_confidence.count)
            return raw_answer, answer, used_hints, confidence
        
        else:
            # generate prompt
            prompt, tip, system_define = self.select_prompt(node.isMultipleChoice(), 'answer')
            if node.context is not None:
                prompt += 'Context: ' + node.context + '\n'
            prompt += 'Question: '
            # prompt += 'Q: Yes or no: '
            prompt += node.question + '\n'
            if node.isMultipleChoice():
                prompt += 'Option: ' + node.add_optionID_toText() + '\n'
            if node.hasHint():
                prompt += 'Hints: ' + node.get_textHints() + '\n'
            prompt += tip + '\n'
            prompt += 'Answer: '
            
            # request chatGPT
            raw_answer = self.request(system_define, prompt)
            
            # post-process for socratic cot
            answer = raw_answer
            if ' answer is: ' in raw_answer:
                # locate position, and the next charactor is the answer
                idx = raw_answer.index(' answer is: ')
                answer = raw_answer[idx+12:]
            elif ' answer is ' in raw_answer:
                idx = raw_answer.index(' answer is ')
                answer = raw_answer[idx+11:]
            elif ' option is: ' in raw_answer:
                # locate position, and the next charactor is the answer
                idx = raw_answer.index(' option is: ')
                answer = raw_answer[idx+12:]
            elif ' option is ' in raw_answer:
                idx = raw_answer.index(' option is ')
                answer = raw_answer[idx+11:]
            answer_parts = answer.split(';')
            
            optionID_part = answer_parts[0].strip() # Answer: A
            if ': ' in optionID_part:
                answer = optionID_part.split(': ')[1].strip() # A
            else:
                answer = optionID_part.strip()
            if node.isMultipleChoice():
                answer = self.get_clean_answer(answer)
            
            pattern = re.compile(r'\[Answer: (.*); Used hints: (.*); Confidence: (.*)\]')
            match = pattern.search(raw_answer)
            if match is not None:
                answer = match.group(1)
                
            hints_part = None
            if len(answer_parts) > 1:
                hints_part = answer_parts[1].strip() # Used hints: 1,2,3
            if hints_part is not None:
                if ': ' in hints_part:
                    used_hints = hints_part.split(': ')[1].strip()
                else:
                    used_hints = hints_part.strip()
            else:
                used_hints = None
                
            confidence_part = None
            if len(answer_parts) > 2:
                confidence_part = answer_parts[2].strip() 
            if confidence_part is not None:
                if 'high' in confidence_part or 'High' in confidence_part:
                    confidence = 'high'
                elif 'middle' in confidence_part or 'Middle' in confidence_part:
                    confidence = 'middle'
                elif 'low' in confidence_part or 'Low' in confidence_part:
                    confidence = 'low'
                else:
                    confidence = 'middle'
            else:
                confidence = 'low'
                
            return raw_answer, answer, used_hints, confidence
            
            
    
    def raise_question(self, node):
        prompt, tip, system_define = self.select_prompt(False, 'raise')
        if node.context is not None:
            prompt += 'Context: ' + node.context + '\n'
        prompt += 'Question: '
        prompt += node.question + '\n'
        if node.isMultipleChoice():
            prompt += 'Option: ' + node.add_optionID_toText() + '\n'
        prompt += tip + '\n'
        if node.hasHint():
            prompt += 'Hints: ' + node.get_textHints() + '\n'
        prompt += 'Deep Questions: '
        
        # request chatGPT
        raw_answer = self.request(system_define, prompt)
        
        # post-process answer
        ls_questions = raw_answer.split('\n')
        ls_questions = [re.sub(r'^\d+\.\s+', '', q) for q in ls_questions]
        ls_questions = ls_questions[:self.num_question] if len(ls_questions) > self.num_question else ls_questions
        return ls_questions
        
    def self_questioning(self, node):
        raw_answer, answer, used_hints, confidence = self.answer_question(node)      
        node.update_answer(answer, used_hints, confidence)

        # raise question
        if (confidence == 'high' or node.isLeaf) and (node.hasHint() or node.depth != 1):
        # if (confidence == 'high' or node.isLeaf) and (node.turn>2 or node.depth != 1):
        # if (confidence == 'high' or node.isLeaf):
            self.log(node, raw_answer)
            return False, answer
        else:
            ls_deepQuestion = self.raise_question(node)
            self.log(node, raw_answer, ls_deepQuestion=ls_deepQuestion)
            return True, ls_deepQuestion
    
    def run(self, node):
        # self questioning
        continue_deeper, output = self.self_questioning(node)
        if continue_deeper: # output is subquestion list
            for deep_question in output:
                deep_node = SocraticNode(deep_question, node.turn, node.depth+1, self.max_turn, self.max_depth, context=node.context)
                deep_answer = self.run(deep_node)
                hint = self.qa2hint(deep_question, deep_answer)
                node.add_hint(hint)
            node.update_turn_num()
            return self.run(node)
            
        else: # output is answer
            return output
        
    def start(self, id, question, options, context=None):
        self.log_path = self.save_dir + '/log/' + str(id) + '.txt'
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        self.root_node = SocraticNode(question, 1, 1, self.max_turn, self.max_depth, context=context, options=options)
        answer = self.run(self.root_node)
        return answer, self.root_node, self.root_node.hints
        
    def log(self, node, raw_answer, ls_deepQuestion=None):
        turn = node.turn
        depth = node.depth
        isLeaf = node.isLeaf
        context = node.context
        question = node.question
        options = None
        if node.isMultipleChoice():
            options = node.add_optionID_toText()
        hints = node.get_textHints()
        raw_answer = raw_answer
        answer = str(node.answer)
        ls_deepQuestion = ls_deepQuestion       
        
        with open(self.log_path, 'a') as f:
            f.write('=========================================================================================================\n')
            f.write('Turn: ' + str(turn) + ', Depth: ' + str(depth) + ', isLeaf: ' + str(isLeaf) + '\n')
            if context is not None:
                f.write('Context:\t' + context + '\n')
            f.write('Question:\t' + question + '\n')
            if options is not None:
                f.write('Options:\t' + options + '\n')
            if len(hints) > 0:
                f.write('Hints:\t' + hints + '\n')
            f.write('Raw Answer:\t' + raw_answer + '\n')
            f.write('Answer:\t' + answer + '\n')
            if ls_deepQuestion is not None:
                f.write('\nDeep Questions:\n')
                for i, deepQuestion in enumerate(ls_deepQuestion):
                    f.write(str(i+1) + '. ' + deepQuestion + '\n')
            f.write('\n')        
            
             
            