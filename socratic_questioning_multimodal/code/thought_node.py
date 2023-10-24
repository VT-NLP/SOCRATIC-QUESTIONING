import json
import openai
import pdb
import random

class ThoughtNode:
    def __init__(self, api, id, question, if_fact_question, img_path, depth, turn_num, max_depth=2, max_turn_num=2, context=None, hint=None, object_regions=None, llm=None):
        self.api = api
        self.llm = llm
        if api is not None:
            openai.api_key = self.api
        
        self.question_id = id
        self.depth = depth
        self.turn_num = turn_num
        self.max_depth = max_depth
        self.max_turn_num = max_turn_num
        
        
        self.question = question
        self.if_fact_question = if_fact_question
        self.img_path = img_path
        self.context = context
        self.hint = hint
        self.sub_question = None
        self.object_regions = object_regions
        
        self.children = []
        self.parent = None
        
    def create_child_node(self, id, deeper_question, if_fact_question):
        child_node = ThoughtNode(self.api, id, deeper_question, if_fact_question, self.img_path, self.depth+1, self.turn_num, self.max_depth, self.max_turn_num, hint=self.hint, llm=self.llm)
        child_node.set_parent(self)
        self.children.append(child_node)
        return child_node
    
    def set_parent(self, parent):
        self.parent = parent
        
            
    def ifRoot(self):
        if self.parent is None:
            return True
        return False
        
    def hasContext(self):
        if self.context is None:
            return False
        return True
    
    def get_context(self):
        if self.context is None:
            raise Exception("Context is not set yet!")
        return self.context
    
    def set_context(self, context):
        self.context = context
        
    def merge_hint(self):
        merged_hint = ''
        j = 1
        for k, v in self.hint.items():
            for i in range(len(v)):
                merged_hint += '(' + str(j) + ') ' + v[i] + '\n'
                j+=1
                
        return merged_hint
    
    def merge_single_hint(self, question, answer):
        if self.llm is None:
            prompt = "Question: " + question + "\nAnswer: " + answer
            response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "system", "content": """You are given a question-and-answer pair, can you help me to merge the question and answer into a statement sentence. If the question or the answer is ambiguous you can just output the token "unknown". If the merged sentence is ambiguous, you can just output the token "unknown". If you can merge the question-and-answer pair, just output the sentence."""},
                                    {"role": "user", "content": prompt},
                            ]
                        )   
            return response['choices'][0]['message']['content']
        else:
            demo =  """You are given a question-and-answer pair, can you help me to merge the question and answer into a statement sentence. If the question or the answer is ambiguous you can just output the token "unknown". If the merged sentence is ambiguous, you can just output the token "unknown". If you can merge the question-and-answer pair, just output the sentence.\n"""
            prompt = demo + "Question: " + question + "\nAnswer: " + answer
            return self.llm.request_vicuna(prompt).strip()
            
    def get_hint(self):
        if self.hint is None:
            raise Exception("Hint is not set yet!")
        else:
            return self.merge_hint()
        
    def get_visual_hint_list(self):
        hint_list = []
        if self.hint['visual'] is None:
            raise Exception("Hint is not set yet!")
        else:
            # merged_hint = ''
            fact_hint_list = self.hint['fact']
            hint = ''
            for i in range(len(self.hint['visual'])):
                for j in range(len(fact_hint_list)):
                    hint += '(' + str(j+1) + ') ' + fact_hint_list[j] + ' '
                hint += '(' + str(j+2) + ') ' + self.hint['visual'][i] + ' '
                hint_list.append(hint)
            return hint_list
    
    def set_hint(self, question, answer, hint_type):
        hint = self.merge_single_hint(question, answer)
        if self.hint is None:
            self.hint = {}
            self.hint['fact'] = []
            self.hint['visual'] = []
        self.hint[hint_type].append(hint)
        # self.hint = list(set(self.hint))
        if self.sub_question is None:
            self.sub_question = [(hint_type, question,answer)]
        self.sub_question.append((hint_type, question,answer))

    def update_turn_num(self):
        self.turn_num += 1
    
    def ifLack(self, output):
        if 'lack of information' in output:
            return True
        if 'Lack of information' in output:
            return True
        if 'cannot answer' in output:
            return True
        if 'unknow' in output:
            return True
        if 'Unknow' in output:
            return True
        return False
        
    def ifDeeper(self):
        if self.depth >= self.max_depth or self.turn_num >= self.max_turn_num:
            return False
        return True
    
    def answer_fact_question(self):        
        if self.llm is None:
            prompt = "Q: " + self.question
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "Imagine you are a polymath familiar with encyclopedias and all kinds of common-sense knowledge. You need to answer a question about some facts or some common-sense knowledge in short sentence."},
                        {"role": "user", "content": "Q: What is human life expectancy in the United States?"},
                        {"role": "assistant", "content": "Human life expectancy in the United States is 78 years."},
                        {"role": "user", "content": "Q: Who was president of the United States in 1955?"},
                        {"role": "assistant", "content": "Dwight D. Eisenhower was president of the United States in 1955."},
                        {"role": "user", "content": "Q: Which party did he belong to?"},
                        {"role": "assistant", "content": "He belonged to the Republican Party."},
                        {"role": "user", "content": "Q: How does a telescope work?"},
                        {"role": "assistant", "content": "Telescopes use lenses or mirrors to focus light and make objects appear closer."},
                        {"role": "user", "content": prompt},
                ]
            )   
            answer = response['choices'][0]['message']['content']
            continue_deeper = False
            
            return answer, continue_deeper
        else:
            demo = "Imagine you are a polymath familiar with encyclopedias and all kinds of common-sense knowledge. You need to answer a question about some facts or some common-sense knowledge in short sentence.\nQ: What is human life expectancy in the United States?\nHuman life expectancy in the United States is 78 years.\nQ: Who was president of the United States in 1955?\nDwight D. Eisenhower was president of the United States in 1955.\nQ: Which party did he belong to?\nHe belonged to the Republican Party.\nQ: How does a telescope work?\nTelescopes use lenses or mirrors to focus light and make objects appear closer.\n"
            prompt = demo + "Q: " + self.question
            answer = self.llm.request_vicuna(prompt).strip()
            continue_deeper = False
            
            return answer, continue_deeper
    
    def answer_visual_question(self, prompts_map, consistency=True):
        def most_frequent(List):
            counter = 0
            num = List[0]
            for i in List:
                curr_frequency = List.count(i)
                if(curr_frequency >= counter):
                    counter = curr_frequency
                    num = i
            if counter == 1:
                num = List[-1]
            return num
    
        if self.ifDeeper():
            tip = prompts_map['tip_q']
            prompt = prompts_map['prompt_q']            
        else:
            tip = prompts_map['tip_a']
            prompt = prompts_map['prompt_a']
            
        if not self.ifRoot():
            if self.ifDeeper():
                tip = prompts_map['tip_dq']
                prompt = prompts_map['prompt_dq']            
            else:
                tip = prompts_map['tip_da']
                prompt = prompts_map['prompt_da']
        
        try:
            prompt += self.get_context()
        except Exception as e:
            pdb.set_trace()
        
        if consistency:
            answer_list = []
            if self.hint is not None:
                if len(self.hint['visual']) > 0:
                    for hint in self.get_visual_hint_list():
                        if self.llm is None:
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                        {"role": "user", "content": prompt + "Important Hints: " + hint + "\n"+"Q: " + self.question + tip + "\nA:"},
                                ]
                            )            
                            answer = response['choices'][0]['message']['content'].strip()
                        else:
                            answer = self.llm.request_vicuna(prompt + "Important Hints: " + hint + "\n"+"Q: " + self.question + tip + "\nA:").strip()
                        answer_list.append(answer)
                        
                    if self.llm is None: 
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "user", "content": prompt + "\n"+"Q: " + self.question + tip + "\nA:"},
                            ]
                        )            
                        answer = response['choices'][0]['message']['content'].strip()
                    else: 
                        answer = self.llm.request_vicuna(prompt + "\n"+"Q: " + self.question + tip + "\nA:").strip()
                    answer_list.append(answer)
                    answer = most_frequent(answer_list)
                    # pdb.set_trace()
                else:
                    prompt += "Important Hints: " + self.get_hint() + "\n"
                    prompt += "Q: " + self.question + tip + "\nA:"
                    
                    if self.llm is None:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                    {"role": "user", "content": prompt},
                            ]
                        )            
                        answer = response['choices'][0]['message']['content'].strip()
                    else:
                        answer = self.llm.request_vicuna(prompt).strip()
                    # pdb.set_trace()
            else:
                prompt += "Q: " + self.question + tip + "\nA:"
                if self.llm is None:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                                {"role": "user", "content": prompt},
                        ]
                    )            
                    answer = response['choices'][0]['message']['content'].strip()
                    # pdb.set_trace()
                else:
                    answer = self.llm.request_vicuna(prompt).strip()
            
            
        continue_deeper = self.ifLack(answer) and self.ifDeeper()
        
        return answer, continue_deeper
    
    def raise_question(self, question_type):
        
        prompt = self.get_context()
        if self.hint is not None:
            prompt = prompt + " Additional Context: " + self.get_hint()
        prompt += "\nQuestion: " + self.question
        prompt += "\nRaise some help questions, not answer the question."
        # pdb.set_trace()
        if question_type == 'visual':
            if self.llm is None:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "Imagine you are a blind but intelligent system. You are given the context of an image and a question about the image. However, the current context is insufficient to answer the question. You should ask me at least two short questions about visual information in the image to help you answer the question. Important notes: do not use pronouns in your generated questions. Each question can only contain one argument. Do not just ask Yes/No questions."},
                            {"role": "user", "content": "Image Caption: two women walking on a sidewalk with an umbrella.\nQuestion: Are the ladies friends?\nRaise some help questions, not answer the question."},
                            {"role": "assistant", "content": "Idea: People with close relationships, such as friends, walk closer.\nAre the two women walking close to each other?\nIdea: The body language between friends will be more intimate, such as hugging, holding hands, etc.\nWhat's the body language of the two women?"},
                            {"role": "user", "content": '''Image Caption: a horse pulling a carriage with two people in it. Additional Context: {'fact': ["People generally use tools like bridles to force horses to work."], 'visual': ["The sun is light, and it is very hot."]}\nQuestion: Does the horse do this because it wants to?\nRaise some help questions, not answer the question.'''},
                            {"role": "assistant", "content": "Idea: When animals are forced to work, they show facial expressions such as anger and sadness.\nWhat is the expression on the horse's face while it's pulling the carriage?\nIdea: Humans often use tools such as bridles to control animals and force them to work.\nWhat type of tools or equipment is being used to control the horse while it pulls the carriage?\nIdea: Horses will be reluctant to work when they are very tired.\nIs the horse sweating or showing any signs of exhaustion?"},
                            {"role": "user", "content": prompt},
                    ]
                )
            else:
                demo = '''Imagine you are a blind but intelligent system. You are given the context of an image and a question about the image. However, the current context is insufficient to answer the question. You should ask me at least two short questions about visual information in the image to help you answer the question. Important notes: do not use pronouns in your generated questions. Each question can only contain one argument. Do not just ask Yes/No questions.\n\nImage Caption: two women walking on a sidewalk with an umbrella.\nQuestion: Are the ladies friends?\nRaise some help questions, not answer the question.\nAre the two women walking close to each other?\nIf the two women are walking side by side?\nWhat's the body language of the two women?\n\nImage Caption: a horse pulling a carriage with two people in it. Additional Context: {'fact': ["People generally use tools like bridles to force horses to work."], 'visual': ["The sun is light, and it is very hot."]}\nQuestion: Does the horse do this because it wants to?\nRaise some help questions, not answer the question.\nWhat is the expression on the horse's face while it's pulling the carriage?\nWhat type of tools or equipment is being used to control the horse while it pulls the carriage?\n Is the horse sweating or showing any signs of exhaustion?\n\n'''
                return self.llm.request_vicuna(demo + prompt).strip()
            
        elif question_type == 'fact':
            if self.llm is None:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {"role": "system", "content": "Imagine you are a blind but intelligent asker only given a question and description of an image. You need to ask me (at most 5) questions about facts or commonsense knowledge to help you get more information about the image to help answer the given question. Important notes: do not use pronouns in your generated questions. Each question can only contain one argument. Do not just ask Yes/No questions."},
                            {"role": "user", "content": "Image Caption: a bowl of oranges in a bowl.\nQuestion: What states are these grown in?\nRaise some help questions, not answer the question."},
                            {"role": "assistant", "content": "Idea: Each US state has different specialties\nIn which state in the USA are oranges grown?\nIdea: Oranges grow in states where the environment is good for them\nWhat environments are suitable for orange growth?"},
                            {"role": "user", "content": "Image Caption: a woman and a boy wearing umbrellas on their heads.\nQuestion: How many watts is that microwave?\nRaise some help questions, not answer the question."},
                            {"role": "assistant", "content": "Idea: Household appliances usually have general standar.\nWhat is the average power of a microwave oven?"},
                            {"role": "user", "content": "Image Caption: a group of people standing on a dock next to a plane.\nQuestion: How many passengers can this plane accommodate?\nRaise some help questions, not answer the question."},
                            {"role": "assistant", "content": "Idea: Different types of aircraft have their general passenger capacity standards.\nWhat is the passenger capacity of a small airliner in general?"},
                            {"role": "user", "content": "Image Caption: a bedroom with a bed and a canopy. Important Hint: the style of room decoration is pink.\nQuestion: Is this a room for a boy or a girl?\nRaise some help questions, not answer the question."},
                            {"role": "assistant", "content": "Idea: Different gender prefers different interior design colors.\nWhat is the generally used interior design color of a boy's and girl's bedrooms?\nIdea: want to know color used in image is preferred by which gender\nIs pink generally prefered by boy or girl?\nIdea: want to know common decoration used in bedroom of a boy\nWhat are the decorations in a boy's bedroom?"},
                            {"role": "user", "content": prompt},
                    ]
                )            
            else:
                demo = '''Imagine you are a blind but intelligent asker only given a question and description of an image. You need to ask me (at most 5) questions about facts or commonsense knowledge to help you get more information about the image to help answer the given question. Important notes: do not use pronouns in your generated questions. Each question can only contain one argument. No same questions allowed. Do not just ask Yes/No questions.\n\nImage Caption: a bowl of oranges in a bowl.\nQuestion: What states are these grown in?\nRaise some help questions, not answer the question. No same questions allowed.\nIn which state in the USA are oranges grown?\nWhat environments are suitable for orange growth?\n\nImage Caption: a piece of bread on a plate. Important Hint: There is mold on the bread surface.\nQuestion: How does the bread taste?\nRaise some help questions, not answer the question. No same questions allowed.\n"What are the characteristics of good-tasting bread?\nIs mold delicious?\nHow to determine if a bread taste good?\n\nImage Caption: a woman and a boy wearing umbrellas on their heads.\nQuestion: How many watts is that microwave?\nRaise some help questions, not answer the question. No same questions allowed.\nWhat is the average power of a microwave oven?\n\nImage Caption: a group of people standing on a dock next to a plane.\nQuestion: How many passengers can this plane accommodate?\nRaise some help questions, not answer the question. No same questions allowed.\nWhat is the passenger capacity of a small airliner in general?\n\nImage Caption: a bedroom with a bed and a canopy. Important Hint: the style of room decoration is pink.\nQuestion: Is this a room for a boy or a girl?\nRaise some help questions, not answer the question. No same questions allowed.\nWhat is the generally used interior design color of a boy's and girl's bedrooms?\nIs pink generally prefered by boy or girl?\nWhat are the decorations in a boy's bedroom?\nWhat are the decorations in a girl's bedroom?\n\n'''
                return self.llm.request_vicuna(demo + prompt).strip()
        else:
            raise ValueError("Question_type should be visual or fact")
        
        return response['choices'][0]['message']['content']
    
    