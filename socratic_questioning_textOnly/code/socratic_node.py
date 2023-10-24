import json
import random
import pdb

class SocraticNode:
    def __init__(self, question, turn, depth, max_turn, max_depth, context=None, options=None):
        self.context = context
        self.question = question
        self.options = options
        self.hints = []
        self.answer = None
        self.turn = turn
        self.depth = depth
        self.max_turn = max_turn
        self.max_depth = max_depth
        self.isLeaf = turn >= max_turn or depth >= max_depth
        
    def isMultipleChoice(self):
        if self.depth==1 and self.options is not None:
            return True
        else:
            return False
    
    def hasHint(self):
        if len(self.hints) == 0:
            return False
        else:
            return True
        
    def add_hint(self, hint):
        self.hints.append(hint)
        
    def get_textHints(self):
        hints = ''
        for i, hint in enumerate(self.hints):
            hints += '(' + str(i+1) + ') ' + hint + '; '   
        if len(hints) > 0:  
            hints = hints[:-2] + '.'
        return hints
    
    def add_optionID_toList(self):
        option_ls = []
        if self.options is None:
            pdb.set_trace()
        for i, option in enumerate(self.options):
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

    def add_optionID_toText(self):
        option_text = str(self.add_optionID_toList())
        return option_text
    
    def update_answer(self, answer, used_hints, confidence):
        self.answer = {
            'answer': answer,
            'used_hints': used_hints,
            'confidence': confidence
        }        
        
    def update_turn_num(self):
        self.turn += 1
        self.isLeaf = self.turn >= self.max_turn or self.depth >= self.max_depth