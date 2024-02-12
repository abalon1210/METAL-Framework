import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd

tasks = ['toxicity_detection', 'news_classification', 'sentiment_analysis', 'question_answering', 'text_summarization', 'information_retrieval']
llms = ['GooglePaLM','Llama2', 'GPT'] 
perturbType= ['GooglePaLM', 'CreatedFunctions','Llama2', 'GPT'] # PaLMAPI,    etc

R_log_path = './Outputs/'
F_log_path = './Outputs/Fairness/'

def MR_Toxicity_Detection(task, target):
    MR_results = [] # Matrix for saving all iterations of MR comparison results
    MR_EF_results = []
    Perturb_qualities= []
    
    task_path = target + task + "/"
    for file_path in os.listdir(task_path):
        log = pd.read_csv(task_path + "/" + file_path)
        
        MR_comparison = [] # A set of MR comparison results
        Perturb_qualities_MR = []
        old_flag = None
        for index, item in log.iterrows():
            # Efficiency
            MR_EF_results.append(item['OriginalTime'] - item['PerturbedTime'])
            
            #Robustness
            flag = item['InputTextID']
            
                
            if type(old_flag) is not str:
                old_flag = flag
                
            if old_flag != flag:
                MR_results.append(MR_comparison[:])
                MR_comparison.clear()
                old_flag = flag
                Perturb_qualities.append(Perturb_qualities_MR[:])
                Perturb_qualities_MR.clear()

            # Perturbation Quality
            pert_sts, pert_sent = PerturbationQuality(item['InputText'].lower(), item['PerturbedText'].lower())
            if pert_sts == -1 and pert_sent == -1:
                Perturb_qualities_MR.append(0.0)
            else:
                val = pert_sts*pert_sent
                if val < 0:
                    val = 0
                Perturb_qualities_MR.append(val)
            
            if type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is not str or item['PerturbedText'] == 'nan':
                MR_comparison.append(-1.0)
                continue
            elif type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is str: 
                MR_comparison.append(-2.0)
                continue
            elif type(item['OriginalOutput']) is str and type(item['PerturbedOutput']) is not str: # None output from perturbed texts by LLMs
                MR_comparison.append(0.1)
                continue
            else:
                org_text = [x.strip() for x in item['OriginalOutput'].lower().split('.')]
                pert_text = [x.strip() for x in item['PerturbedOutput'].lower().split('.')]
                
                if len(org_text) >= 2:
                    org_text = org_text[:-1]
                    
                if len(pert_text) >= 2:
                    pert_text = pert_text[:-1]
                
                o_flag = 0 # Equivalence check for the two Toxicity detection outputs are the same
                if ('yes' in org_text[0] and 'yes' in pert_text[0]) or ('no' in org_text[0] and 'no' in pert_text[0]):
                    o_flag += 0.5
                else: # The two outputs are different
                    MR_comparison.append(0.0)
                    continue
                flag = 1
                for i in range(1,len(org_text)):
                    if i >= len(pert_text):
                        break
                    if org_text[i] != pert_text[i]: # All reasons are matched
                        flag = 0
                if flag == 1:
                    o_flag += 0.5
                MR_comparison.append(o_flag)            
        MR_results.append(MR_comparison[:])
        Perturb_qualities.append(Perturb_qualities_MR[:])
            
    return MR_results, MR_EF_results, Perturb_qualities

def MR_News_Classification(task, target):
    MR_results = [] # Matrix for saving all iterations of MR comparison results
    MR_EF_results = []
    Perturb_qualities = []
    
    task_path = target + task + "/"
    for file_path in os.listdir(task_path):
        log = pd.read_csv(task_path + "/" + file_path)

        old_flag = None
        MR_comparison = [] # A set of MR comparison results
        Perturb_qualities_MR = []
        for index, item in log.iterrows():
            # Efficiency
            MR_EF_results.append(item['OriginalTime'] - item['PerturbedTime'])
            
            #Robustness
            flag = item['InputTextID']
            
            if type(old_flag) is not str:
                old_flag = flag
                
            if old_flag != flag:
                MR_results.append(MR_comparison[:])
                MR_comparison.clear()
                old_flag = flag
                Perturb_qualities.append(Perturb_qualities_MR[:])
                Perturb_qualities_MR.clear()
                
            # Perturbation Quality
            pert_sts, pert_sent = PerturbationQuality(item['InputText'].lower(), item['PerturbedText'].lower())
            if pert_sts == -1 and pert_sent == -1:
                Perturb_qualities_MR.append(0.0)
            else:
                val = pert_sts*pert_sent
                if val < 0:
                    val = 0
                Perturb_qualities_MR.append(val)

            if type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is not str or item['PerturbedText'] == 'nan':
                MR_comparison.append(-1.0)
                continue
            elif type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is str:
                MR_comparison.append(-2.0)
                continue
            elif type(item['OriginalOutput']) is str and type(item['PerturbedOutput']) is not str: # None output from perturbed texts by LLMs
                MR_comparison.append(0.1)
                continue
            else:
                org_text = item['OriginalOutput'].lower()
                pert_text = item['PerturbedOutput'].lower()
                
                if org_text == pert_text:
                    MR_comparison.append(1.0)
                else: 
                    MR_comparison.append(0.0)                  
        MR_results.append(MR_comparison[:])
        Perturb_qualities.append(Perturb_qualities_MR[:])     
            
    return MR_results, MR_EF_results, Perturb_qualities
            

def MR_Sentiment_Analysis(task, target):
    MR_results = [] # Matrix for saving all iterations of MR comparison results
    MR_EF_results = []
    Perturb_qualities = []
    
    task_path = target + task + "/"
    for file_path in os.listdir(task_path):
        log = pd.read_csv(task_path + "/" + file_path)
        
        old_flag = None
        MR_comparison = [] # A set of MR comparison results
        Perturb_qualities_MR = []
        for index, item in log.iterrows():
            # Efficiency
            MR_EF_results.append(item['OriginalTime'] - item['PerturbedTime'])
            
            #Robustness
            flag = item['InputTextID']
            
            if type(old_flag) is not str:
                old_flag = flag
                
            if old_flag != flag:
                MR_results.append(MR_comparison[:])
                MR_comparison.clear()
                old_flag = flag
                Perturb_qualities.append(Perturb_qualities_MR[:])
                Perturb_qualities_MR.clear()
                
            # Perturbation Quality
            if type(item['InputText']) is not str or type(item['PerturbedText']) is not str:
                pert_sts = -1
                pert_sent = -1
            else:
                pert_sts, pert_sent = PerturbationQuality(item['InputText'].lower(), item['PerturbedText'].lower())
            if pert_sts == -1 and pert_sent == -1:
                Perturb_qualities_MR.append(0.0)
            else:
                if item['PerturbationID'] == 'Replacing words with their antonyms':
                    val = (1-pert_sts)*pert_sent
                    if val < 0:
                        val = 0
                    Perturb_qualities_MR.append(val)
                else:
                    val = pert_sts*pert_sent
                    if val < 0:
                        val = 0
                    Perturb_qualities_MR.append(val)
                
            
            if type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is not str  or item['PerturbedText'] == 'nan':
                MR_comparison.append(-1.0)
                continue
            elif type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is str: 
                MR_comparison.append(-2.0)
                continue
            elif type(item['OriginalOutput']) is str and type(item['PerturbedOutput']) is not str: # None output from perturbed texts by LLMs
                MR_comparison.append(0.1)
                continue
            else:
                org_text = item['OriginalOutput'].lower()
                pert_text = item['PerturbedOutput'].lower()
                
                if item['PerturbationID'] == 'Replacing words with their antonyms': # Antonym substitution -> Inequivalence
                    if org_text != pert_text:
                        MR_comparison.append(1.0)
                    else:
                        MR_comparison.append(0.0)
                else: # Other perturbation functions -> Equivalence
                    if org_text == pert_text:
                        MR_comparison.append(1.0)
                    else: 
                        MR_comparison.append(0.0)            
        MR_results.append(MR_comparison[:])
        Perturb_qualities.append(Perturb_qualities_MR[:])
                       
            
    return MR_results, MR_EF_results, Perturb_qualities

def MR_Question_Answering(task, target):
    MR_results = [] # Matrix for saving all iterations of MR comparison results
    MR_EF_results = []
    Perturb_qualities = []
    
    task_path = target + task + "/"
    for file_path in os.listdir(task_path):
        if ".DS" in file_path:
            continue
        log = pd.read_csv(task_path + "/" + file_path)
        
        old_flag = None
        MR_comparison = [] # A set of MR comparison results
        Perturb_qualities_MR = []
        for index, item in log.iterrows():
            # Efficiency
            MR_EF_results.append(item['OriginalTime'] - item['PerturbedTime'])
            
            #Robustness
            flag = item['InputTextID']
            
            if type(old_flag) is not str:
                old_flag = flag
                
            if old_flag != flag:
                MR_results.append(MR_comparison[:])
                MR_comparison.clear()
                old_flag = flag
                Perturb_qualities.append(Perturb_qualities_MR[:])
                Perturb_qualities_MR.clear()
                
            # Perturbation Quality
            if type(item['InputText']) is not str or type(item['PerturbedText']) is not str:
                pert_sts = -1
                pert_sent = -1
            else:
                pert_sts, pert_sent = PerturbationQuality(item['InputText'].lower(), item['PerturbedText'].lower())
            if pert_sts == -1 and pert_sent == -1:
                Perturb_qualities_MR.append(0.0)
            else:
                if item['PerturbationID'] == 'Replacing words with their antonyms':
                    val = (1-pert_sts)*pert_sent
                    if val < 0:
                        val = 0
                    Perturb_qualities_MR.append(val)
                else:
                    val = pert_sts*pert_sent
                    if val < 0:
                        val = 0
                    Perturb_qualities_MR.append(val)
                
            if type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is not str or item['PerturbedText'] == 'nan':
                MR_comparison.append(-1.0)
                continue
            elif type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is str: 
                MR_comparison.append(-2.0)
                continue
            elif type(item['OriginalOutput']) is str and type(item['PerturbedOutput']) is not str: # None output from perturbed texts by LLMs
                MR_comparison.append(0.1)
                continue
            else:
                org_text = item['OriginalOutput'].lower()
                pert_text = item['PerturbedOutput'].lower()
                                
                if item['PerturbationID'] == 'Replacing words with their antonyms': # Antonym substitution -> Inequivalence
                    if org_text != pert_text:
                        MR_comparison.append(1.0)
                    else:
                        MR_comparison.append(0.0)
                else: # Other perturbation functions -> Equivalence
                    if org_text in pert_text or pert_text in org_text or org_text == pert_text:
                        MR_comparison.append(1.0)
                    else: 
                        MR_comparison.append(0.0)                
        MR_results.append(MR_comparison[:])
        Perturb_qualities.append(Perturb_qualities_MR[:])   
    
    return MR_results, MR_EF_results, Perturb_qualities

def MR_Text_Summarization(task, target):
    MR_results = [] # Matrix for saving all iterations of MR comparison results
    MR_EF_results = []
    Perturb_qualities = []
    
    task_path = target + task + "/"
    for file_path in os.listdir(task_path):
        log = pd.read_csv(task_path + "/" + file_path)

        old_flag = None
        MR_comparison = [] # A set of MR comparison results
        Perturb_qualities_MR = []
        for index, item in log.iterrows():
            # Efficiency
            MR_EF_results.append(item['OriginalTime'] - item['PerturbedTime'])
            
            #Robustness
            flag = item['InputTextID']
            
            if type(old_flag) is not str:
                old_flag = flag
                
            if old_flag != flag:
                MR_results.append(MR_comparison[:])
                MR_comparison.clear()
                old_flag = flag
                Perturb_qualities.append(Perturb_qualities_MR[:])
                Perturb_qualities_MR.clear()
                
            # Perturbation Quality
            if type(item['InputText']) is not str or type(item['PerturbedText']) is not str:
                pert_sts = -1
                pert_sent = -1
            else:
                pert_sts, pert_sent = PerturbationQuality(item['InputText'].lower(), item['PerturbedText'].lower())
            if pert_sts == -1 and pert_sent == -1:
                Perturb_qualities_MR.append(0.0)
            else:
                if item['PerturbationID'] == 'Replacing words with their antonyms':
                    val = (1-pert_sts)*pert_sent
                    if val < 0:
                        val = 0
                    Perturb_qualities_MR.append(val)
                else:
                    val = pert_sts*pert_sent
                    if val < 0:
                        val = 0
                    Perturb_qualities_MR.append(val)
                
            if type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is not str  or item['PerturbedText'] == 'nan':
                MR_comparison.append(-1.0)
                continue
            elif type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is str: 
                MR_comparison.append(-2.0)
                continue
            elif type(item['OriginalOutput']) is str and type(item['PerturbedOutput']) is not str: # None output from perturbed texts by LLMs
                MR_comparison.append(0.1)
                continue
            else:
                org_text = [x.strip() for x in item['OriginalOutput'].lower().split('.')]
                pert_text = [x.strip() for x in item['PerturbedOutput'].lower().split('.')]
                
                if len(org_text) >= 2:
                    org_text = org_text[:-1]
                    
                if len(pert_text) >= 2:
                    pert_text = pert_text[:-1]

                sts = []
                for t1, t2 in zip(org_text, pert_text):
                    sts.append(np.inner(embed([t1]), embed([t2]))[0][0])
                MR_comparison.append(sum(sts)/len(sts))                
        MR_results.append(MR_comparison[:])  
        Perturb_qualities.append(Perturb_qualities_MR[:])
    
    return MR_results, MR_EF_results, Perturb_qualities

def MR_Information_Retrieval(task, target):
    MR_results_STS = [] # Matrix for saving all iterations of MR comparison results
    MR_results_MSRD = []
    MR_EF_results = []
    Perturb_qualities = []
    
    task_path = target + task + "/"
    for file_path in os.listdir(task_path):
        log = pd.read_csv(task_path + "/" + file_path)
        
        MR_comparison_STS = [] # A set of MR comparison results
        MR_comparison_MSRD = []
        Perturb_qualities_MR = []
        
        old_flag = None
        for index, item in log.iterrows():# Efficiency
            MR_EF_results.append(float(item['OriginalTime']) - float(item['PerturbedTime']))
            
            #Robustness
            flag = item['InputTextID']
            
            if type(old_flag) is not str:
                old_flag = flag
                
            if old_flag != flag:
                MR_results_STS.append(MR_comparison_STS[:])
                MR_results_MSRD.append(MR_comparison_MSRD[:])
                MR_comparison_STS.clear()
                MR_comparison_MSRD.clear()
                old_flag = flag
                Perturb_qualities.append(Perturb_qualities_MR[:])
                Perturb_qualities_MR.clear()
                
            # Perturbation Quality
            if type(item['InputText']) is not str or type(item['PerturbedText']) is not str  or item['PerturbedText'] == 'nan':
                Perturb_qualities_MR.append(-1.0)
            else:
                pert_sts, pert_sent = PerturbationQuality(item['InputText'].lower(), item['PerturbedText'].lower())
                if (pert_sts == -1 and pert_sent == -1) or (pert_sts == 0 and pert_sts == 0):
                    Perturb_qualities_MR.append(0.0)
                else:
                    if item['PerturbationID'] == 'Replacing words with their antonyms':
                        val = (1-pert_sts)*pert_sent
                        if val < 0:
                            val = 0
                        Perturb_qualities_MR.append(val)
                    else:
                        val = pert_sts*pert_sent
                        if val < 0:
                            val = 0
                        Perturb_qualities_MR.append(val)
            
            if type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is not str:
                MR_comparison_STS.append(-1.0)
                MR_comparison_MSRD.append(-1.0)
                continue
            elif type(item['OriginalOutput']) is not str and type(item['PerturbedOutput']) is str: 
                MR_comparison_STS.append(-2.0)
                MR_comparison_MSRD.append(-2.0)
                continue
            elif type(item['OriginalOutput']) is str and type(item['PerturbedOutput']) is not str: # None output from perturbed texts by LLMs
                MR_comparison_STS.append(0.1)
                MR_comparison_MSRD.append(0.1)
                continue
            else:
                org_text = [x.strip()[3:] for x in item['OriginalOutput'].lower().splitlines()]
                pert_text = [x.strip()[3:] for x in item['PerturbedOutput'].lower().splitlines()]

                MR_comparison_STS.append(STS_Ranking(org_text, pert_text))
                MR_comparison_MSRD.append(Most_Similar_Ranking_Diff(org_text, pert_text))
                    
        MR_results_STS.append(MR_comparison_STS[:])
        MR_results_MSRD.append(MR_comparison_MSRD[:])
        Perturb_qualities.append(Perturb_qualities_MR[:])

    return MR_results_STS, MR_results_MSRD, MR_EF_results, Perturb_qualities

def STS_Ranking(org_text, pert_text): # Ranking distance metric for the average STS on each matched ranking
    sts = []
    for t1, t2 in zip(org_text, pert_text):
        sts.append(np.inner(embed([t1]), embed([t2]))[0][0])
        
    return sum(sts) / len(sts)

def Most_Similar_Ranking_Diff(org_text, pert_text): # Ranking distance for the average rank-index differences between the most similar items
    ranking_diffs = []
    for id1, t1 in enumerate(org_text):
        max_sts = -1.0
        max_id2 = -1
        for id2, t2 in enumerate(pert_text):
            sts = np.inner(embed([t1]), embed([t2]))[0][0]
            if sts  > max_sts:
                max_id2 = id2
                max_sts = sts
        ranking_diffs.append(abs(max_id2 - id1))

    return sum(ranking_diffs) / len(ranking_diffs)


def MR_ND_EQ(task, target):
    task_path = target + task + "/"
    
    diff_list = []
    iter_list = []
    for file_path in os.listdir(task_path):
        log = pd.read_csv(task_path + "/" + file_path)

        org_outputs = []
        old_flag = None
        for index, item in log.iterrows(): # Dictionary for counting the number of outputs from the same original inputs
            flag = item['InputTextID']
            
            if type(old_flag) is not str or old_flag != flag:
                if item['OriginalOutput'] is not str:
                    org_outputs.append(item['OriginalOutput'])
                else:
                    org_outputs.append(item['OriginalOutput'].lower())
                old_flag = flag
    
        iter_list.append(org_outputs[:])
        
    for i in range(100): # of inputs
        temp = 0
        flag = iter_list[0][i]
        for j in range(1,len(iter_list)):
            if i >= len(iter_list[j]):
                break
            if (type(flag) is str or type(flag) is list) and flag != iter_list[j][i]:
                temp += 1
        diff_list.append(temp/len(iter_list)) # # of iteration
            
    return sum(diff_list) / len(diff_list)

def MR_ND_SEM_EQ(task, target):
    task_path = target + task + "/"
    
    diff_list = []
    iter_list = []
    for file_path in os.listdir(task_path):
        if ".DS" in file_path:
            continue
        log = pd.read_csv(task_path + "/" + file_path)

        org_outputs = []
        old_flag = None
        for index, item in log.iterrows(): # Dictionary for counting the number of outputs from the same original inputs
            flag = item['InputTextID']
            
            if type(old_flag) is not str or old_flag != flag:
                if type(item['OriginalOutput']) is not str:
                    org_text = item['OriginalOutput']
                    org_outputs.append(org_text)
                elif task == "information retrieval":
                    org_text = [x.strip()[3:] for x in item['OriginalOutput'].lower().splitlines()]
                    org_outputs.append(org_text[:])    
                else:
                    org_text = [x.strip() for x in item['OriginalOutput'].lower().splitlines()]
                    org_outputs.append(org_text[:])
                old_flag = flag
    
        iter_list.append(org_outputs[:])
        
    for i in range(100): # of inputs
        temp = []
        
        f_index = -1
        for j in range(0,len(iter_list)):
            if i >= len(iter_list[j]):
                break
            if type(iter_list[j][i]) is list:
                f_index = j
        if f_index < 0:
            if i >= len(iter_list[j]):
                flag = None
            else:
                flag = iter_list[0][i]
        else:
            flag = iter_list[f_index][i]
        
        for j in range(len(iter_list)): # of iterations
            if i >= len(iter_list[j]):
                break
            if j == f_index:
                continue
            if type(flag) is not list:
                if type(iter_list[j][i]) is not list:
                    temp.append(0.0)
                else:
                    temp.append(1.0)
            else:
                if type(iter_list[j][i]) is not list:
                    temp.append(1.0)
                    continue
                if len(flag) > 1 and len(iter_list[j][i]) > 1:
                    val_list = []
                    for s1, s2 in zip(flag, iter_list[j][i]):
                        val_list.append(1 - np.inner(embed([s1]), embed([s2]))[0][0])
                    val = sum(val_list) / len(val_list)
                else:
                    val = 1 - np.inner(embed(flag), embed(iter_list[j][i]))[0][0]
                if val <= 0:
                    temp.append(0)
                else:
                    temp.append(val)
        if len(temp) < 1:
            diff_list.append(0)
        else:
            diff_list.append(sum(temp)/len(temp))
    
    return sum(diff_list) / len(diff_list)
    
def PerturbationQuality(org_text, pert_text):
    if type(org_text) is not str or type(pert_text) is not str: # Input error
        return -1, -1
    
    org_text_list = [x.strip() for x in org_text.lower().split('.')]
    pert_text_list =  [x.strip() for x in pert_text.lower().split('.')]
        
    if len(org_text_list) > 1:
        org_text_list = org_text_list[:-1]
        
        temp = []
        for txt in org_text_list:
            if len(txt) < 1:
                continue
            temp.append(txt.replace("'","")[:])
        org_text_list = temp[:]
    if len(pert_text_list) > 1:
        pert_text_list = pert_text_list[:-1]
        
        temp = []
        for txt in pert_text_list:
            if len(txt) < 1:
                continue
            temp.append(txt.replace("'","")[:])
        pert_text_list = temp[:]
    
    temp = len(pert_text_list) / len(org_text_list)
    if temp > 1:
        sentence_diff = 1
    else:
        sentence_diff = temp
    
    id_flag = True
    for s1, s2 in zip(org_text_list, pert_text_list):
        if len(s1) > 1 and len(s2) > 1:
            while True:
                if s2[0] == '\'' or s2[0] == "'":
                    temp = np.inner(embed([s1]), embed([s2]))[0][0]
                    s2 = s2[1:]
                else:
                    break
            while True:
                if s2[-1] == '\'' or s2[-1] == "'":
                    temp = np.inner(embed([s1]), embed([s2]))[0][0]
                    s2 = s2[:-1]
                else:
                    break
        if s1 != s2:
            id_flag = False
            
    if id_flag and sentence_diff == 0: # No perturbation applied
        return 0, 0
    
    context_sim = None
    count = 0
    context_temp_sim = 0

    for s1, s2 in zip(org_text_list, pert_text_list):
        context_temp_sim += np.inner(embed([s1]), embed([s2]))[0][0]
        count += 1

    context_sim = context_temp_sim / count
    
    if context_sim >= 0.98: # No perturbation applied
        return 0, 0
    
    return context_sim, sentence_diff
    
def main(run_option):
    ND_Results = []

    for llm in llms:
        if run_option == "Robustness":
            for perturb in perturbType:
                target = R_log_path + perturb + "_to_" + llm + "/"
                print(target)
                for count, task in enumerate(tasks):

                    if count == 0: # toxicity detection -> 0: not matched, 0.5: matched but different reasons, 1: fully matched
                        print("TD R MR Analysis")
                        ND_Results.append(MR_ND_EQ(task, target))
                        R_TD_MR_Results, EF_TD_Results, R_TD_Perturb = MR_Toxicity_Detection(task, target)

                    elif count == 1: # News Classification
                        print("NC R MR Analysis")
                        ND_Results.append(MR_ND_EQ(task, target))
                        R_NC_MR_Results, EF_NC_Results, R_NC_Perturb = MR_News_Classification(task, target)
                
                    elif count == 2: # Sentiment Analysis
                        print("SA R MR Analysis")
                        ND_Results.append(MR_ND_EQ(task, target))
                        R_SA_MR_Results, EF_SA_Results, R_SA_Perturb = MR_Sentiment_Analysis(task, target)

                    elif count == 3: # Question Answering
                        print("QA R MR Analysis")
                        ND_Results.append(MR_ND_SEM_EQ(task, target))
                        R_QA_MR_Results, EF_QA_Results, R_QA_Perturb = MR_Question_Answering(task, target)

                    elif count == 4: # Text Summarization
                        print("TS R MR Analysis")
                        ND_Results.append(MR_ND_SEM_EQ(task, target))
                        R_TS_MR_Results, EF_TS_Results, R_TS_Perturb = MR_Text_Summarization(task, target)

                    else: # Information Retrieval
                        print("IR MR Analysis")
                        ND_Results.append(MR_ND_SEM_EQ(task, target))
                        R_IR_MR_Results_STS, R_IR_MR_Results_MSRD, EF_IR_Results, R_IR_Perturb = MR_Information_Retrieval(task, target)
                
                with pd.ExcelWriter(R_log_path + perturb + "_to_" + llm + ".xlsx") as writer:
                    # Robustness 
                    pd.DataFrame(R_TD_MR_Results).to_excel(writer, sheet_name="R_TD")
                    pd.DataFrame(R_SA_MR_Results).to_excel(writer, sheet_name="R_SA")
                    pd.DataFrame(R_NC_MR_Results).to_excel(writer, sheet_name="R_NC")
                    pd.DataFrame(R_QA_MR_Results).to_excel(writer, sheet_name="R_QA")
                    pd.DataFrame(R_TS_MR_Results).to_excel(writer, sheet_name="R_TS")
                    pd.DataFrame(R_IR_MR_Results_STS).to_excel(writer, sheet_name="R_IR_STS")
                    pd.DataFrame(R_IR_MR_Results_MSRD).to_excel(writer, sheet_name="R_IR_MSRD")
                    
                    # Non-Determinisms
                    pd.DataFrame(ND_Results).to_excel(writer, sheet_name="ND")
                    
                    # Efficiency
                    pd.DataFrame(EF_TD_Results).to_excel(writer, sheet_name="EF_TD")
                    pd.DataFrame(EF_SA_Results).to_excel(writer, sheet_name="EF_SA")
                    pd.DataFrame(EF_NC_Results).to_excel(writer, sheet_name="EF_NC")
                    pd.DataFrame(EF_QA_Results).to_excel(writer, sheet_name="EF_QA")
                    pd.DataFrame(EF_TS_Results).to_excel(writer, sheet_name="EF_TS")
                    pd.DataFrame(EF_IR_Results).to_excel(writer, sheet_name="EF_IR")
                    
                     # Perturb Quality
                    pd.DataFrame(R_TD_Perturb).to_excel(writer, sheet_name="P_TD")
                    pd.DataFrame(R_SA_Perturb).to_excel(writer, sheet_name="P_SA")
                    pd.DataFrame(R_NC_Perturb).to_excel(writer, sheet_name="P_NC")
                    pd.DataFrame(R_QA_Perturb).to_excel(writer, sheet_name="P_QA")
                    pd.DataFrame(R_TS_Perturb).to_excel(writer, sheet_name="P_TS")
                    pd.DataFrame(R_IR_Perturb).to_excel(writer, sheet_name="P_IR")
                
        elif run_option == "Fairness":
            target = F_log_path + llm + "/"
            print(target)
            for count, task in enumerate(tasks):

                if count == 0: # toxicity detection -> 0: not matched, 0.5: matched but different reasons, 1: fully matched
                    print("TD F MR Analysis")
                    ND_Results.append(MR_ND_EQ(task, target))
                    F_TD_MR_Results, EF_TD_Results, F_TD_Perturb  = MR_Toxicity_Detection(task, target)

                elif count == 2: # Sentiment Analysis
                    print("SA F MR Analysis")
                    ND_Results.append(MR_ND_EQ(task, target))
                    F_SA_MR_Results, EF_SA_Results, F_SA_Perturb  = MR_Sentiment_Analysis(task, target)

                elif count == 3: # Question Answering
                    print("QA F MR Analysis")
                    ND_Results.append(MR_ND_SEM_EQ(task, target))
                    F_QA_MR_Results, EF_QA_Results, F_QA_Perturb  = MR_Question_Answering(task, target)
            
            with pd.ExcelWriter(F_log_path + llm + ".xlsx") as writer:
                # Robustness 
                pd.DataFrame(F_TD_MR_Results).to_excel(writer, sheet_name="F_TD")
                pd.DataFrame(F_SA_MR_Results).to_excel(writer, sheet_name="F_SA")
                pd.DataFrame(F_QA_MR_Results).to_excel(writer, sheet_name="F_QA")
                
                # Non-Determinisms
                pd.DataFrame(ND_Results).to_excel(writer, sheet_name="ND")
                
                # Efficiency
                pd.DataFrame(EF_TD_Results).to_excel(writer, sheet_name="EF_TD")
                pd.DataFrame(EF_SA_Results).to_excel(writer, sheet_name="EF_SA")
                pd.DataFrame(EF_QA_Results).to_excel(writer, sheet_name="EF_QA")
                
                 # Perturb Quality
                pd.DataFrame(F_TD_Perturb).to_excel(writer, sheet_name="P_TD")
                pd.DataFrame(F_SA_Perturb).to_excel(writer, sheet_name="P_SA")
                pd.DataFrame(F_QA_Perturb).to_excel(writer, sheet_name="P_QA")
    
    
model_path = './EvaluationModule/STS_Models/universal-sentence-encoder_4/'
model_u = hub.KerasLayer(handle=model_path)

def embed(input):
    return model_u(input)

main("Fairness") # "Fairness", "Robustness"