from functools import reduce
from operator import or_

"""Constants.

This file contains several constants.

"""

# Tasks.
NONE_TASK = 0
INTENTION_TASK = 1 << 0
TEXT_TASK = 1 << 1
RECOMMEND_TASK = 1 << 2
KNOWLEDGE_STYLETIP_SUBTASK = 1 << 3
KNOWLEDGE_ATTRIBUTE_SUBTASK = 1 << 4
KNOWLEDGE_CELEBRITY_SUBTASK = 1 << 5
KNOWLEDGE_SUBTASKS = [KNOWLEDGE_STYLETIP_SUBTASK,
                      KNOWLEDGE_ATTRIBUTE_SUBTASK,
                      KNOWLEDGE_CELEBRITY_SUBTASK]
KNOWLEDGE_TASK = reduce(or_, KNOWLEDGE_SUBTASKS)

TASK_STR = {
    INTENTION_TASK: 'intention',
    TEXT_TASK: 'text',
    RECOMMEND_TASK: 'recommend',
    KNOWLEDGE_STYLETIP_SUBTASK: 'knowledge_styletip',
    KNOWLEDGE_ATTRIBUTE_SUBTASK: 'knowledge_attribute',
    KNOWLEDGE_CELEBRITY_SUBTASK: 'knowledge_celebrity'
}

TASK_ID = {task_str: task_id for task_id, task_str in TASK_STR.items()}

# Modes.
NONE_MODE = 0
TRAIN_MODE = 1 << 0
VALID_MODE = 1 << 1
TEST_MODE = 1 << 2
ALL_MODE = -1

MODE_STR = {
    TRAIN_MODE: 'train',
    VALID_MODE: 'valid',
    TEST_MODE: 'test'
}

MODE_ID = {mode_str: mode_id for mode_id, mode_str in MODE_STR.items()}

# Speakers.
USER_SPEAKER = 0
SYS_SPEAKER = 1

# Special tokens.
SOS_TOKEN = '</s>'
EOS_TOKEN = '</e>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

SOS_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3

SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]

PAD_VALUE_ID = 0

# Others.
DIALOG_PROC_PRINT_FREQ = 100
