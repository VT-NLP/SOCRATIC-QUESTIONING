import json

class PromptMap:
    def __init__(self):
        self.tips_dict = {
            'vqa':{
                'tip_q': ' (If the information is not enough to answer the question, answer \"lack of information\")',
                'tip_a': ' (Must return an answer. The answer should be 1 word (maximum 2 words). If you are not sure, you can guess the most plausible answer)',
            },     
            'okvqa': {
                'tip_q': ' (If the information is not enough to answer the question, answer \"lack of information\")',
                'tip_a': ' (Must return an answer. The final answer should be 1 or 2 words (maximum 2 words). If you are not sure, you can guess the most plausible answer)',
            },     
            'aokvqa': {
                'tip_q': ' (If the information is not enough to answer the question, answer \"lack of information\")',
                'tip_a': ' (Must return an answer. The answer should be 1 word (maximum 2 words). If you are not sure, you can guess the most plausible answer)',
            },
            'hatefulmeme': {
                'tip_q': ' (Answer \"yes\" or \"no\" at the end. If the information is not enough to decide if the image contains hateful intent, answer \"lack of information\")',
                'tip_a': ' (Answer \"yes\" or \"no\" at the end. If you are not sure, you can guess the answer)',
            },
            'snlive': {
                'tip_q': ' (Choose an answer from the choices. If the information is not enough to determine the answer, answer "lack of information". The output can only be one of ["yes", "no", "lack of information"].)',
                'tip_a': ' (Choose an answer from the choices. The answer can only be one of ["yes", "no"].)',
            },     
            'vcr': {
                'tip_q': ' (Choose an answer ID from the choices to answer the question. If the information is not enough to answer the question, answer "lack of information". The output can only be one of [0, 1, 2, 3, lack of information].)',
                'tip_a': ' (Choose an answer ID from the choices to answer the question. The ID can only be one of [0, 1, 2, 3].)',
            },
        }
        
        self.prompts_dict = {
            'without_tool':{
                'without_inner_mono':{
                    'vqa':{
                        'prompt_q': "Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question.\n\nImage Caption: a man holding a dog on his back.\nImportant Hints: (1) dogs usually use mouth to catch objects (2) the popular game people play with dog is frisbee (3) the man is holding a frisbee\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hints 1,2,3 are useful. The answer is: mouth\n\nImage Caption: a bowl of oranges in a bowl.\nQ: What states are these grown in?"+self.tips_dict['okvqa']['tip_q']+"\nA: lack of information\n\nImage Caption: A desk with four computers and a phone.\nImportant Hints: (1) Desk is generally used for work and study (2) computers and phone are generally used for working like business or tech\nQ: What is this desk used for?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hint 2 is useful. The answer is: work\n\nImage Caption: a busy city street with many people walking around.\nQ: Why might someone go to this place?"+self.tips_dict['okvqa']['tip_q']+"\nA: shop\n\nImage Caption: a statue of two women sitting on a bench.\nQ: Who designed the statues?"+self.tips_dict['okvqa']['tip_q']+"\nA: artist\n\nImage Caption: a boy playing tennis.\nQ: Is this boy a professional player or still in high school?"+self.tips_dict['okvqa']['tip_q']+"\nA: lack of information\n\nImage Caption: a bathroom with a toilet and a sink.\nImportant Hints: (1) toilet could be used by both man and woman (2) there is a razor near the sink\nQ: Who leaves a toilet like this?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hint 2 is useful.The answer is: man\n\n",
                        'prompt_a': "Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question.\nImage Caption: a man holding a dog on his back.\nImportant Hints: (1) dogs usually use mouth to catch objects (2) the popular game people play with dog is frisbee (3) the man is holding a frisbee\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hints 1,2,3 are useful. The answer is: mouth\n\nImage Caption: a desk with four computers and a phone.\nImportant Hints: (1) Desk is generally used for work and study (2) computers and phone are generally used for working like business or tech\nQ: What is this desk used for?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hint 2 is useful. The answer is: work\n\nImage Caption: a busy city street with many people walking around.\nQ: Why might someone go to this place?"+self.tips_dict['okvqa']['tip_a']+"\nA: shop\n\nImage Caption: a statue of two women sitting on a bench.\nQ: Who designed the statues?"+self.tips_dict['okvqa']['tip_a']+"\nA: artist\n\nImage Caption: a bathroom with a toilet and a sink.\nImportant Hints: (1) toilet could be used by both man and woman (2) there is a razor near the sink\nQ: Who leaves a toilet like this?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hint 2 is useful. The answer is: man\n\n",
                    },
                    'okvqa':{
                        'prompt_q': "Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question.\n\nImage Caption: a man holding a dog on his back.\nImportant Hints: (1) dogs usually use mouth to catch objects (2) the popular game people play with dog is frisbee (3) the man is holding a frisbee\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hints 1,2,3 are useful. The answer is: mouth\n\nImage Caption: a bowl of oranges in a bowl.\nQ: What states are these grown in?"+self.tips_dict['okvqa']['tip_q']+"\nA: lack of information\n\nImage Caption: A desk with four computers and a phone.\nImportant Hints: (1) Desk is generally used for work and study (2) computers and phone are generally used for working like business or tech\nQ: What is this desk used for?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hint 2 is useful. The answer is: work\n\nImage Caption: a busy city street with many people walking around.\nQ: Why might someone go to this place?"+self.tips_dict['okvqa']['tip_q']+"\nA: shop\n\nImage Caption: a statue of two women sitting on a bench.\nQ: Who designed the statues?"+self.tips_dict['okvqa']['tip_q']+"\nA: artist\n\nImage Caption: a boy playing tennis.\nQ: Is this boy a professional player or still in high school?"+self.tips_dict['okvqa']['tip_q']+"\nA: lack of information\n\nImage Caption: a bathroom with a toilet and a sink.\nImportant Hints: (1) toilet could be used by both man and woman (2) there is a razor near the sink\nQ: Who leaves a toilet like this?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hint 2 is useful.The answer is: man\n\n",
                        'prompt_a': "Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question.\nImage Caption: a man holding a dog on his back.\nImportant Hints: (1) dogs usually use mouth to catch objects (2) the popular game people play with dog is frisbee (3) the man is holding a frisbee\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hints 1,2,3 are useful. The answer is: mouth\n\nImage Caption: a desk with four computers and a phone.\nImportant Hints: (1) Desk is generally used for work and study (2) computers and phone are generally used for working like business or tech\nQ: What is this desk used for?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hint 2 is useful. The answer is: work\n\nImage Caption: a busy city street with many people walking around.\nQ: Why might someone go to this place?"+self.tips_dict['okvqa']['tip_a']+"\nA: shop\n\nImage Caption: a statue of two women sitting on a bench.\nQ: Who designed the statues?"+self.tips_dict['okvqa']['tip_a']+"\nA: artist\n\nImage Caption: a bathroom with a toilet and a sink.\nImportant Hints: (1) toilet could be used by both man and woman (2) there is a razor near the sink\nQ: Who leaves a toilet like this?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hint 2 is useful. The answer is: man\n\n",
                    },
                    'aokvqa':{
                        'prompt_q': "Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question.\n\nImage Caption: a man holding a dog on his back.\nImportant Hints: (1) dogs usually use mouth to catch objects (2) the popular game people play with dog is frisbee (3) the man is holding a frisbee\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hints 1,2,3 are useful. The answer is: mouth\n\nImage Caption: a bowl of oranges in a bowl.\nQ: What states are these grown in?"+self.tips_dict['okvqa']['tip_q']+"\nA: lack of information\n\nImage Caption: A desk with four computers and a phone.\nImportant Hints: (1) Desk is generally used for work and study (2) computers and phone are generally used for working like business or tech\nQ: What is this desk used for?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hint 2 is useful. The answer is: work\n\nImage Caption: a busy city street with many people walking around.\nQ: Why might someone go to this place?"+self.tips_dict['okvqa']['tip_q']+"\nA: shop\n\nImage Caption: a statue of two women sitting on a bench.\nQ: Who designed the statues?"+self.tips_dict['okvqa']['tip_q']+"\nA: artist\n\nImage Caption: a boy playing tennis.\nQ: Is this boy a professional player or still in high school?"+self.tips_dict['okvqa']['tip_q']+"\nA: lack of information\n\nImage Caption: a bathroom with a toilet and a sink.\nImportant Hints: (1) toilet could be used by both man and woman (2) there is a razor near the sink\nQ: Who leaves a toilet like this?"+self.tips_dict['okvqa']['tip_q']+"\nA: Hint 2 is useful.The answer is: man\n\n",
                        'prompt_a': "Imagine you are a blind but intelligent question answering system. You are asked a visual question about an image. I will provide you the caption of the image and some useful visual hints. Please use your best judgement to answer the visual question.\nImage Caption: a man holding a dog on his back.\nImportant Hints: (1) dogs usually use mouth to catch objects (2) the popular game people play with dog is frisbee (3) the man is holding a frisbee\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hints 1,2,3 are useful. The answer is: mouth\n\nImage Caption: a desk with four computers and a phone.\nImportant Hints: (1) Desk is generally used for work and study (2) computers and phone are generally used for working like business or tech\nQ: What is this desk used for?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hint 2 is useful. The answer is: work\n\nImage Caption: a busy city street with many people walking around.\nQ: Why might someone go to this place?"+self.tips_dict['okvqa']['tip_a']+"\nA: shop\n\nImage Caption: a statue of two women sitting on a bench.\nQ: Who designed the statues?"+self.tips_dict['okvqa']['tip_a']+"\nA: artist\n\nImage Caption: a bathroom with a toilet and a sink.\nImportant Hints: (1) toilet could be used by both man and woman (2) there is a razor near the sink\nQ: Who leaves a toilet like this?"+self.tips_dict['okvqa']['tip_a']+"\nA: Hint 2 is useful. The answer is: man\n\n",
                    },
                    'vcr':{
                        'prompt_q': "Imagine you are a blind but intelligent reasoning system. You are asked to do some reasoning based on a given image. I will provide you the caption of the image and some useful visual hints. However, the caption and visual hints may not be accurate. Please use your best judgement to select one answer from the choices.\n\nThe text on the image are: \"a bowl of oranges.\"\nQ: What states are these grown in? Choices: ['0: california', '1: new york', '2: washington', '3: texas']"+self.tips_dict['aokvqa']['tip_q']+"\nA: lack of information\n\nThe text on the image are: \"a bedroom with a bed and a canopy.\"\nQ: Is this a room for a boy or a girl? Choices: ['0: girl', '1: boy']"+self.tips_dict['aokvqa']['tip_q']+"\nA:  0\n\nThe text on the image are: \"a boy playing tennis.\"\nQ: Is this boy a professional player or still in high school? Choices: ['0: professional player', '1: in high school']"+self.tips_dict['aokvqa']['tip_q']+"\nA:  lack of information\n\nThe text on the image are: \"a display case filled with lots of donuts in a commercial kitchen.\"\nQ: Is this in a home kitchen or commercial kitchen? Choices: ['0: home', '1: commercial']"+self.tips_dict['aokvqa']['tip_q']+"\nA:  1\n\nThe text on the image are: \"a statue of two women sitting on a bench.\"\nQ: Who designed the statues? Choices: ['0: researcher', '1: singer', '2: dancer', '3: artists']"+self.tips_dict['aokvqa']['tip_q']+"\nA:  3\n\nThe text on the image are: \"a man holding a dog on his back.\"\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding? Choices: ['0: leg', '1: head', '2: mouth', '3: foot']"+self.tips_dict['aokvqa']['tip_q']+"\nA:  2\n\n",
                        'prompt_a': "Imagine you are a blind but intelligent reasoning system. You are asked to do some reasoning based on a given image. I will provide you the caption of the image and some useful visual hints. However, the caption and visual hints may not be accurate. Please use your best judgement to select one answer from the choices.\n\nThe text on the image are: \"a bowl of oranges.\"\nQ: What states are these grown in? Choices: ['0: california', '1: new york', '2: washington', '3: texas']"+self.tips_dict['aokvqa']['tip_a']+"\nA:  0\n\nThe text on the image are: \"a bedroom with a bed and a canopy.\"\nQ: Is this a room for a boy or a girl? Choices: ['0: girl', '1: boy']"+self.tips_dict['aokvqa']['tip_a']+"\nA:  0\n\nThe text on the image are: \"a statue of two women sitting on a bench.\"\nQ: Who designed the statues? Choices: ['0: researcher', '1: singer', '2: dancer', '3: artists']"+self.tips_dict['aokvqa']['tip_a']+"\nA:  3\n\nThe text on the image are: \"a man holding a dog on his back.\"\nQ: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding? Choices: ['0: leg', '1: head', '2: mouth', '3: foot']"+self.tips_dict['aokvqa']['tip_a']+"\nA:  2\n\n",
                    },
                    'hatefulmeme':{
                        'prompt_q': f"Imagine you are a blind but intelligent hateful content detection system. You are given the caption and the text written on an image. You need to consider both the text and the caption to deicide if the image contains any hateful intention. There are some additional hints about the image. {self.tips_dict['hatefulmeme']['tip_q']}\n\n",
                        'prompt_a': f"Imagine you are a blind but intelligent hateful content detection system. You are given the caption and the text written on an image. You need to consider both the text and the caption to deicide if the image contains any hateful intention. There are some additional hints about the image. {self.tips_dict['hatefulmeme']['tip_a']}\n\n" 
                    },
                    'snlive':{
                        'prompt_q': f"Imagine you are a blind but intelligent image content determination system. You are given the caption and a statement of one image. You need to determine if the image entails the statement. There are some additional hints about the image. {self.tips_dict['snlive']['tip_q']}\n\nImage Caption: A collage of one person climbing a cliff\nQ: Can you conclude the text from the image? Text: The person is on the swings. Choices: [\"yes\", \"no\"]\nA: No\n\nImage Caption: a group of people climbing up a rock wall\nQ: Can you conclude the text from the image? Text: A person is climbing. Choices: [\"yes\", \"no\"]\nA: yes\n\nImage Caption: a group of people climbing up a rock wall\nQ: Can you conclude the text from the image? Text: Some people are falling to their deaths. Choices: [\"yes\", \"no\"]\nA: lack of information\n\n",
                        'prompt_a': f"Imagine you are a blind but intelligent image content determination system. You are given the caption and a statement of one image. You need to determine if the image entails the statement. There are some additional hints about the image. {self.tips_dict['snlive']['tip_a']}\n\nImage Caption: A collage of one person climbing a cliff\nQ: Can you conclude the text from the image? Text: The person is on the swings. Choices: [\"yes\", \"no\"]\nA: No\n\nImage Caption: a group of people climbing up a rock wall\nQ: Can you conclude the text from the image? Text: A person is climbing. Choices: [\"yes\", \"no\"]\nA: yes\n\n" 
                    }
                }
            }  
        }
        
    def get_map(self, task_type):
        tip_q, tip_a = self.get_tips(task_type)
        prompt_q, prompt_a = self.get_prompts(task_type)
        tip_dq, tip_da = self.get_tips('okvqa')
        prompt_dq, prompt_da = self.get_prompts('okvqa')
        
        return {'tip_q': tip_q, 'tip_a': tip_a, 'prompt_q': prompt_q, 'prompt_a': prompt_a, 'tip_dq': tip_dq, 'tip_da': tip_da, 'prompt_dq': prompt_dq, 'prompt_da': prompt_da}
    
    def get_tips(self, task_type):            
        tips = self.tips_dict[task_type]
        tip_q = tips['tip_q']
        tip_a = tips['tip_a']
        
        return tip_q, tip_a
    
    def get_prompts(self, task_type):
        prompts = self.prompts_dict['without_tool']['without_inner_mono'][task_type]
        prompt_q = prompts['prompt_q']
        prompt_a = prompts['prompt_a']
        
        return prompt_q, prompt_a