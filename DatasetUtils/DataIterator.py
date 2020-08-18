import torch
from torchtext import data
from torchtext.data import BucketIterator
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def MultiWoZ(data_folder = './Dataset/MultiWoZ/',task = 'dialogue', batch_size = 2, max_length=100):
    def contexttoks(x): return x.split()[-max_length:]
    def toks(x): return x.split()
    context_id = data.Field()
    # multi class
    all_topics = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    # Single class
    filename = None
    utterance_index = None
    # Sequential Data
    TEXT = data.Field(sequential=True, fix_length = max_length ,eos_token = '<eor>', init_token = '<sos>',pad_token = '<pad>', tokenize = contexttoks)
    #target = data.Field()
    # single class
    target_length = data.Field(sequential=False)
    dialogue_position = data.Field(sequential=False)
    # MultiLabel
    repeating_info = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    # single class
    current_dialogue_topic = data.Field(sequential=False) #build_vocab
    # MultiLabel
    recent_slots = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    recent_values = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    # single class
    number_local_info = data.Field(sequential=False)
    # MultiLabel
    all_values = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    all_slots = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    # Single Class
    number_all_info = data.Field(sequential=False)
    num_overlap_slots = data.Field(sequential=False)
    num_topics = data.Field(sequential=False)
    is_multi_task = data.Field(sequential=False)
    # Multilabel
    entity_slots = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    entity_values = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    action = data.Field(pad_token = '' , unk_token = None, tokenize = toks)

    if task == 'dialogue':
        train = data.TabularDataset(path=os.path.join(data_folder,'MultiWoZ_train.csv'), format='csv',fields=[('context_id',None),('AllTopics',None),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('length',None),('UtteranceLoc',dialogue_position) ])
        valid = data.TabularDataset(path=os.path.join(data_folder,'MultiWoZ_valid.csv'), format='csv',fields=[('context_id',None),('AllTopics',None),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('length',None),('UtteranceLoc',dialogue_position) ])
        test =  data.TabularDataset(path=os.path.join(data_folder,'MultiWoZ_test.csv'), format='csv',fields=[('context_id',None),('AllTopics',None),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('length',None),('UtteranceLoc',dialogue_position) ])

    else:
        train = data.TabularDataset(path=os.path.join(data_folder,'MultiWoZ_train.csv'), format='csv',fields=[('context_id',None),('AllTopics',all_topics),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('Responselength',target_length),('UtteranceLoc',dialogue_position),('RepeatInfo',repeating_info),('RecentTopic',current_dialogue_topic),\
    ('RecentSlots',recent_slots),('RecentValues',recent_values),('NumRecentInfo',number_local_info),('AllValues',all_values),('AllSlots',all_slots),('NumAllInfo',number_all_info),\
    ('NumRepeatInfo',num_overlap_slots),('NumAllTopics',num_topics),('IsMultiTask',is_multi_task),('EntitySlots',entity_slots),('EntityValues',entity_values),('ActionSelect',action)])
        valid = data.TabularDataset(path=os.path.join(data_folder,'MultiWoZ_valid.csv'), format='csv',fields=[('context_id',None),('AllTopics',all_topics),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('Responselength',target_length),('UtteranceLoc',dialogue_position),('RepeatInfo',repeating_info),('RecentTopic',current_dialogue_topic),\
    ('RecentSlots',recent_slots),('RecentValues',recent_values),('NumRecentInfo',number_local_info),('AllValues',all_values),('AllSlots',all_slots),('NumAllInfo',number_all_info),\
    ('NumRepeatInfo',num_overlap_slots),('NumAllTopics',num_topics),('IsMultiTask',is_multi_task),('EntitySlots',entity_slots),('EntityValues',entity_values),('ActionSelect',action)])
        test = data.TabularDataset(path=os.path.join(data_folder,'MultiWoZ_test.csv'), format='csv',fields=[('context_id',None),('AllTopics',all_topics),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('Responselength',target_length),('UtteranceLoc',dialogue_position),('RepeatInfo',repeating_info),('RecentTopic',current_dialogue_topic),\
    ('RecentSlots',recent_slots),('RecentValues',recent_values),('NumRecentInfo',number_local_info),('AllValues',all_values),('AllSlots',all_slots),('NumAllInfo',number_all_info),\
    ('NumRepeatInfo',num_overlap_slots),('NumAllTopics',num_topics),('IsMultiTask',is_multi_task),('EntitySlots',entity_slots),('EntityValues',entity_values),('ActionSelect',action)])

    train_iter, valid_iter ,test_iter = BucketIterator.splits((train, valid, test),batch_size = batch_size, sort_key=lambda x: x.UtteranceLoc,device = device)
    TEXT.build_vocab(train,min_freq=3)
    context_id.build_vocab(train,valid,test)
    dialogue_position.build_vocab(train,valid,test)
    return train_iter, valid_iter, test_iter, TEXT.vocab.stoi['<pad>'], len(TEXT.vocab), TEXT.vocab.itos, context_id.vocab.itos

def PersonaChat(data_folder = './Dataset/PersonaChat/',task = 'dialogue', batch_size = 2, max_length=100):
    def contexttoks(x): return x.split()[-max_length:]
    def toks(x): return x.split()
    context_id = data.Field()
    # Single class
    filename = None
    utterance_index = None
    # Sequential Data
    TEXT = data.Field(sequential=True, fix_length = max_length ,eos_token = '<eor>', init_token = '<sos>',pad_token = '<pad>', tokenize = contexttoks)
    #target = data.Field()
    # single class
    target_length = data.Field(sequential=False)
    dialogue_position = data.Field(sequential=False)
    # MultiLabel
    personal_info = data.Field(pad_token = '' , unk_token = None, tokenize = toks, is_target = True)
    wordCont = data.Field(sequential=False)

    if task == 'dialogue':
        train = data.TabularDataset(path=os.path.join(data_folder,'PersonaChat_train.csv'), format='csv',fields=[('context_id',context_id),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('length',None) ])
        valid = data.TabularDataset(path=os.path.join(data_folder,'PersonaChat_valid.csv'), format='csv',fields=[('context_id',context_id),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('length',None) ])

    else:
        train = data.TabularDataset(path=os.path.join(data_folder,'PersonaChat_train.csv'), format='csv',fields=[('context_id',None),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('Responselength',target_length),('UtteranceLoc',dialogue_position),('PersonalInfo',personal_info),('WordCont',wordCont)])
        valid = data.TabularDataset(path=os.path.join(data_folder,'PersonaChat_valid.csv'), format='csv',fields=[('context_id',None),('filename', None),\
    ('UtteranceIndex',None),('Context',TEXT),('Target',TEXT),('Responselength',target_length),('UtteranceLoc',dialogue_position),('PersonalInfo',personal_info),('WordCont',wordCont)])

    train_iter, valid_iter = BucketIterator.splits((train, valid),batch_size = batch_size, sort_key=lambda x: len(x.Target),device = device)
    TEXT.build_vocab(train,min_freq=3)
    context_id.build_vocab(train,valid)
    return train_iter, valid_iter, TEXT.vocab.stoi['<pad>'], len(TEXT.vocab), TEXT.vocab.itos, context_id.vocab.itos
