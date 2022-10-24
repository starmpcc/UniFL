import pandas as pd
import numpy as np
import re
from operator import itemgetter
from itertools import groupby
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

quant=20
vocab = {
    '[PAD]' : 0,
    '[CLS]' : 1,
    '[SEP]' : 2,
    '[MASK]' : 3
}

vocab['TB_0'] = 4
start_idx = 5
for qb in range(1, quant+1):
    vocab[f'TB_{qb}'] = start_idx
    start_idx+=1

number_token_list = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 119] 


def eicu_med_revise(df):
    df['split'] = df['dosage'].apply(lambda x: str(re.sub(',', '',str(x))).split())
    def second(x):
        try:
            if len(pd.to_numeric(x))>=2:
                x = x[1:]
            return x
        except ValueError:
            return x

    df['split'] = df['split'].apply(second).apply(lambda s:' '.join(s))
    punc_dict = str.maketrans('', '', '.-')
    df['uom'] = df['split'].apply(lambda x: re.sub(r'[0-9]', '', x))
    df['uom'] = df['uom'].apply(lambda x: x.translate(punc_dict)).apply(lambda x: x.strip())
    df['uom'] = df['uom'].apply(lambda x: ' ' if x=='' else x)
    
    def hyphens(s):
        if '-' in str(s):
            s = str(s)[str(s).find("-")+1:]
        return s
    df['value'] = df['split'].apply(hyphens)
    df['value'] = df['value'].apply(lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)])
    df['value'] = df['value'].apply(lambda x: x[-1] if len(x)>0 else x)
    df['value'] = df['value'].apply(lambda d: str(d).replace('[]',' '))
    df = df.drop('split',axis=1)
    df = df.drop('dosage',axis=1)
    return df


def eicu_inf_revise(df):
    df['split'] = df['drugname'].apply(lambda x: str(x).rsplit('(', maxsplit=1))
    def addp(x):
        if len(x)==2:
            x[1] = '(' + str(x[1])
        return x

    df['split'] = df['split'].apply(addp)
    df['split']=df['split'].apply(lambda x: x +[' '] if len(x)<2 else x)

    df['drugname'] = df['split'].apply(lambda x: x[0])
    df['uom'] = df['split'].apply(lambda x: x[1])
    df['uom'] = df['uom'].apply(lambda s: s[s.find("("):s.find(")")+1])

    toremove = ['()','', '(Unknown)', '(Scale B)', '(Scale A)',  '(Human)', '(ARTERIAL LINE)']

    df['uom'] = df['uom'].apply(lambda uom: ' ' if uom in toremove else uom)
    df = df.drop('split',axis=1)
    
    testing = lambda x: (str(x)[-1].isdigit()) if str(x)!='' else False
    code_with_num = list(pd.Series(df.drugname.unique())[pd.Series(df.drugname.unique()).apply(testing)==True])
    add_unk = lambda s: str(s)+' [UNK]' if s in code_with_num else s
    df['drugname'] = df['drugname'].apply(add_unk)
    
    return df


def name_dict(df, code_dict, column_name):
    key = code_dict['ITEMID']
    value = code_dict['LABEL']
    code_dict = dict(zip(key,value))
    df[column_name] = df[column_name].map(code_dict)
    df[column_name] = df[column_name].map(str)
    
    return df

def ID_time_filter_eicu(df, icu):
   
    df = df[df['ID'].isin(icu['ID'])] # 'ID' filter
    time_fil= df[(df['TIME'] > 0) &
                      (df['TIME'] < 60*12)
                ]
    return time_fil

def ID_time_filter_mimic(df, icu):
    icu['INTIME+12hr'] = icu['INTIME'] + pd.Timedelta(12, unit="h")
    df = df[df['ID'].isin(icu['ID'])]# ID filter
    df['ID'] = df['ID'].astype('int')
    df['TIME'] = pd.to_datetime(df['TIME'])
  
    df = df.merge(icu[['ID', 'INTIME', 'INTIME+12hr']], on='ID', how='left').reset_index(drop=True)
    time_fil= df[(df['TIME'] > df['INTIME']) &
                      (df['TIME'] < df['INTIME+12hr'])
                ]
    time_fil['TIME'] = (time_fil['TIME'] - time_fil['INTIME']).astype('timedelta64[m]')
    time_fil.drop(columns=['INTIME', 'INTIME+12hr'], inplace=True)
    time_fil.reset_index(drop=True, inplace=True)
    return time_fil

def columns_upper(df):
    df.columns = [x.upper() for x in df.columns]
    return

def codeemb_event_merge(df, table_name):

    target_cols =  [ col for col in df.columns if col not in ['ID','TIME', 'time_bucket', 'time_gap', 'TABLE_NAME', 'ORDER']]
    
    df['event'] = df.apply(lambda x: [x[col] for col in target_cols if x[col] != ' '], axis=1)
    df['type'] = df.apply(lambda x: [table_name+ '_' + col for col in target_cols if x[col] !=' '], axis=1)

    df['event_token'] = df.apply(lambda x : x['event'] + [x['time_bucket']], axis=1)
    df['type_token'] = df.apply(lambda x : x['type'] + ['[TIME]'], axis=1)               
   
    return df

def buckettize_categorize(df, src, numeric_dict, table_name, quant):
    code = numeric_dict[src][table_name]['code']

    type_code = df[code].dtype
    for value_target in numeric_dict[src][table_name]['value']: 
        if value_target in df.columns:
            numeric = df[pd.to_numeric(df[value_target], errors='coerce').notnull()]
            numeric[value_target]= numeric[value_target].astype('float')
            not_numeric = df[pd.to_numeric(df[value_target], errors='coerce').isnull()]

            # buckettize
            numeric = buckettize(numeric, code, value_target, quant)


            numeric[value_target] = 'B_' + numeric[value_target].astype('str')
            df = pd.concat([numeric, not_numeric], axis=0)

    for cate_target in numeric_dict[src][table_name]['cate']:
        if cate_target in df.columns:
            df = categoritize(df, cate_target)
    df[code] = df[code].astype(type_code)

    df.fillna(' ', inplace=True)
    df.replace('nan', ' ', inplace=True)
    return df


def make_dpe(target, number_token_list, integer_start=6):
    if type(target) is not list:
        return None
    elif target ==[]:
        return []
    else:
        dpe = [1]*len(target) # dpe token 1 for the plain text token
        scanning = [pos for pos, char in enumerate(target) if char in number_token_list]

        #grouping
        ranges = []
        for k,g in groupby(enumerate(scanning),lambda x:x[0]-x[1]):
            group = (map(itemgetter(1),g))
            group = list(map(int,group))
            ranges.append((group[0],group[-1]))

        # making dpe     
        dpe_group_list = []
        for (start, end) in ranges:
            group = target[start:end+1]
            digit_index = [pos for pos, char in enumerate(group) if char == number_token_list[-1]] #digit_token
            assert len(digit_index) < 3, "More than 3 digit index in sing group"
            if len(digit_index)==2:
                # ex) 1. 0 2 5. 
                if digit_index[0] == 0:
                    group= group[1:]
                    digit_index = digit_index[1:]
                    start=start+1
            # case seperate if digit or integer only
            if len(digit_index)== 0:
                dpe_group = [integer_start+len(group)-i for i in range(len(group))]
            else:
                # 있으면 소수점 기준으로 왼쪽 오른 walk
                dpe_int = [integer_start-1+len(group[:digit_index[0]])-i+1 for i in range(len(group[:digit_index[0]]))]
                dpe_digit = [i+2 for i in range(len(group[digit_index[0]:]))]
                dpe_group = dpe_int + dpe_digit
            dpe_group_list.append(((start,end), dpe_group))

        for (start, end), dpe_group in dpe_group_list:
            dpe[start:end+1] = dpe_group

        return dpe   

 
def categoritize(df, col_name):
    df[col_name] = df[col_name].map(lambda x: stringrize(x, col_name))
    return df

def buckettize(df, code, target_value, quant):
    df[target_value] = df.groupby([code])[target_value].transform(lambda x: x.rank(method = 'dense'))
   
    df[target_value]= df.groupby([code])[target_value].transform(lambda x: q_cut(x,quant))
    return df

def q_cut(x, cuts):

    unique_var = len(np.unique([i for i in x]))
    nunique = len(pd.qcut(x, min(unique_var, cuts), duplicates = 'drop').cat.categories)
    output = pd.qcut(x, min(unique_var,cuts), labels= range(1, min(nunique, cuts)+1), duplicates = 'drop')
    return output


def stringrize(x, col_name):
    if not (x =='nan' or x==pd.isnull(x)):
        return col_name + '_' + str(x)
    else:
        return ' '

def digit_split(digits : str):
    return [' '.join(d) for d in digits]

def isnumeric(text):
    '''returns True if string s is numeric'''    
            
    return all(s in "0123456789." for s in text) or any(s in "0123456789" for s in text)

def digit_split_in_text(text : str):
    join_list = []
    split = text.split()
    new_split = []
    for text in split:
        if not all(s in "0123456789." for s in text) and any(s in "0123456789" for s in text):
            for i, t in enumerate(text):
                if isnumeric(t):
                    idx = i
            new_split += [text[:idx+1], text[idx+1:]]
        else:
            new_split.append(text)
    split = new_split
   
    for i, d in enumerate(split):
        if isnumeric(d):
            while d.count('.') > 1:
                target = d.rfind('.')
                if target  == (len(d)-1) :
                    d = d[:target]
                else:
                    d = d[:target] + d[(target+1):]
            join_list.append(digit_split(d))

        else:
            join_list.append([d])

    return ' '.join(sum(join_list, []))

def split(word):
    return [char for char in word]

    #split and round
def round_digits(digit : str or float or int):
    if isinstance(digit, str):
        return digit_split_in_text(digit)
    elif digit is np.NAN:
        return ' '
    elif isinstance(digit, float):
        return " ".join(split(str(round(digit, 4))))
    elif isinstance(digit, int):
        return " ".join(split(str(digit)))
    elif isinstance(digit, np.int64):
        return str(digit)
    else: 
        return digit

def text_digit_round(text_list):
    if '.' in text_list:
        decimal_point =  text_list.index('.')
        if len(text_list[decimal_point:])> 5:
            return text_list[:decimal_point+5]
        else:
            return text_list
    else:
        return text_list
   