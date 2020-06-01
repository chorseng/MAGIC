#!/usr/bin/python3

"""Preprocess data.

Usage:
    python preprocess_data.py [--<task>=[<mode>]]

"""

import argparse
from itertools import chain
from typing import List, Tuple, Dict, Set

from constant import MODE_STR, NONE_MODE, MODE_ID
from constant import TASK_STR, TASK_ID
from constant import KNOWLEDGE_TASK, KNOWLEDGE_SUBTASKS
from dataset import RawData
from dataset.tidy_data import generate_tidy_data_file

# Constants.
TASKS: List[str] = list(TASK_STR.values())
MODES: List[str] = list(MODE_STR.values())
RAW_DATA_ARG = 'raw_data'


def parse_cmd() -> Dict[str, List[str]]:
    """Parse commandline parameters.

    Returns:
        Dict[str, List[str]]: Parse result.

    """

    # Definition of argument parser.
    parser = argparse.ArgumentParser(
        description='Preprocess data. Specify multiple TASKS and modes at '
                    'the time may be faster than process them one by one.'
    )

    for task in TASKS:
        parser.add_argument(
            '--{}'.format(task),
            nargs='*',
            choices=MODES,
            help='{} task and its modes (train/valid/test).'.format(task)
        )
    parser.add_argument(
        '--{}'.format(RAW_DATA_ARG),
        nargs='*',
        choices=MODES,
        help='raw data of modes (train/valid/test).'
    )

    # Namespace -> Dict
    parse_res: Dict[str, List[str]] = vars(parser.parse_args())
    return parse_res


def standardize_parse_result(
        cmd_args: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Standardize parsing result.

    Note:
        not specify modes == all modes
        But the mode of TASKS not shown is EMPTY! (EXCEPT NO ARGUMENTS)
        No arguments: all TASKS and all modes
        For example: `--intention` means `--intention train valid test`

    Args:
        cmd_args (Dict[str, List[str]]): Commandline arguments.

    Returns:
        Tuple[Dict[str, List[str]], Set[str]]: Standardized parsing result
        and total modes.

    """

    def remove_none_elements():
        for _task, _modes in cmd_args.items():
            if _modes is not None and not _modes:
                cmd_args[_task] = MODES
            elif _modes is None:
                cmd_args[_task] = []

    if cmd_args[RAW_DATA_ARG] is not None:
        for task, modes in cmd_args.items():
            if task != RAW_DATA_ARG and modes is not None:
                raise ValueError('raw_data and tasks are mutual exclusion.')
        total_modes = MODES
        remove_none_elements()
    else:
        remove_none_elements()
        total_modes: Set[str] = set(chain.from_iterable(cmd_args.values()))

    cmd_args.pop(RAW_DATA_ARG)

    # Special: no task is specified -> all task.
    if not total_modes:
        for task in TASKS:
            cmd_args[task] = MODES
        total_modes = MODES
    return cmd_args, total_modes


def main():
    # Parse commandline parameters and standardize.
    parse_result: Dict[str, List[str]] = parse_cmd()
    parse_result, total_modes = standardize_parse_result(parse_result)
    print('Dataset will be processed: {}'.format(parse_result))
    print('Modes will be processed: {}'.format(total_modes))

    # Modes: Set[str] -> int
    raw_data_mode: int = NONE_MODE
    for mode in total_modes:
        raw_data_mode |= MODE_ID[mode]

    # Get necessary raw data.
    raw_data = RawData(raw_data_mode)

    # Generate tidy data file.
    for task, modes in parse_result.items():
        for mode in modes:
            if TASK_ID[task] == KNOWLEDGE_TASK:
                for subtask in KNOWLEDGE_SUBTASKS:
                    generate_tidy_data_file(raw_data, subtask, MODE_ID[mode])
            else:
                generate_tidy_data_file(raw_data, TASK_ID[task], MODE_ID[mode])


if __name__ == '__main__':
    main()
