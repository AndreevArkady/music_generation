from catboost import CatBoostClassifier
import os
from typing import List
import coloredlogs
import argparse
import json
import logging
import tqdm
from multiprocessing import Pool, context
from dynamics import harmony_seq_pipeline
from kp_corpus_process import prepare_dataset_roots, prepare_dataset_mode


best_params_roots = {
    'n_estimators': 685,
    'learning_rate': 0.07557142211700456,
    'depth': 7,
    'l2_leaf_reg': 62.37934704452352,
    'bootstrap_type': 'Bayesian',
    'random_strength': 1.1488785147899944e-08,
    'bagging_temperature': 1.5481239273448555,
    'od_type': 'Iter',
    'od_wait': 49
}

best_params_mode = {
    'n_estimators': 290,
    'learning_rate': 0.04774974078860072,
    'depth': 10,
    'l2_leaf_reg': 0.10285856054142818,
    'bootstrap_type': 'Bayesian',
    'random_strength': 0.0025811780488595813,
    'bagging_temperature': 0.13879907325154095,
    'od_type': 'IncToDec',
    'od_wait': 28
}

def get_args(default='.') -> argparse.Namespace:
    """Get arguments.
    
    Args:
        default (str) : default path for input_folder and output_folder
    Returns:
        argparse.Namespace : parsed args
    """
    default_pool_num = 25
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_folder',
        default=default,
        type=str,
        help='MIDI file input folder',
    )
    parser.add_argument(
        '-f',
        '--file_name',
        default='',
        type=str,
        help='input MIDI file name',
    )
    parser.add_argument(
        '-o',
        '--output_folder',
        default=default,
        type=str,
        help='MIDI file output folder',
    )
    parser.add_argument(
        '-p',
        '--pool_num',
        default=default_pool_num,
        type=int,
        help='number of processes for harmony labeling',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose log output',
    )
    return parser.parse_args()


def walk(folder_name: str) -> List[str]:
    """Walks through files in folder.
    
    Args:
        folder_name (str) : name of the folder

    Returns:
        list : sorted list of file names in the folder
    """
    files = []
    for path, _, all_files in os.walk(folder_name, followlinks=True):
        for file_name in all_files:
            endname = file_name.split('.')[-1].lower()
            if endname == 'tsv':
                files.append(os.path.join(path, file_name))
    return files


def harmony_labeling_custom(task):
    file_name, _, model_roots, model_mode = task
    base_name = os.path.basename(file_name)

    try:
        seq = harmony_seq_pipeline(file_name, model_roots, model_mode)
    except Exception:
        return file_name, os.path.dirname(base_name), base_name, None
    return file_name, os.path.dirname(base_name), base_name, seq


if __name__ == '__main__':

    args = get_args()
    args.output_folder = os.path.abspath(args.output_folder)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    logger = logging.getLogger(__name__)

    logger.handlers = []
    logfile = '{0}/tokenize.log'.format(args.output_folder)
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=logfile,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    coloredlogs.install(level='INFO', logger=logger, isatty=True)


    model_roots = CatBoostClassifier(**best_params_roots, logging_level='Silent')
    X_roots, y_roots = prepare_dataset_roots()
    model_roots.fit(X_roots, y_roots)


    model_mode = CatBoostClassifier(**best_params_mode, logging_level='Silent')
    X_mode, y_mode = prepare_dataset_mode()
    model_mode.fit(X_mode, y_mode)


    output_json_name = os.path.join(args.output_folder, 'files_result.json')

    files_result = {}

    if args.file_name:
        all_names = [args.file_name]
        args.input_folder = os.path.dirname(args.file_name)
    else:
        all_names = walk(args.input_folder)

    tasks = [(file_name, args, model_roots, model_mode) for file_name in all_names]
    print(len(tasks), len(all_names))

    # num_processes = args.pool_num
    # with Pool(num_processes) as p:
    #     res = list(tqdm.tqdm(p.imap(harmony_labeling_custom, tasks), total=len(all_names)))

    res = []
    num_processes = args.pool_num
    max_time = 1200
    pbar = tqdm.tqdm(total=len(tasks))
    while tasks:
        with Pool(num_processes) as pool:
            futures_res = pool.imap(harmony_labeling_custom, tasks.copy())
            while tasks:
                task = tasks.pop(0)
                pbar.update(1)
                try:
                    future_res = futures_res.next(timeout=max_time)
                    res.append(future_res)
                except context.TimeoutError:
                    logger.info(
                        'stuck on file {0}, timeout err, skip'.format(task[0]),
                    )
                    break
    pbar.close()

    for r in res:

        file_name, new_output_folder, base_name, d = r

        if d is not None and len(d) != 0:
            files_result['{0}/{1}'.format(new_output_folder, base_name)] = []
            files_result[
                '{0}/{1}'.format(new_output_folder, base_name)
            ].append(d)
        else:
            logger.info(
                'cannot find harmony labeling of song {0}, skip this file'.format(file_name),
            )

    logger.info(len(files_result))
    with open(os.path.join(args.output_folder, 'files_result.json'), 'w') as fp:
        json.dump(files_result, fp)
