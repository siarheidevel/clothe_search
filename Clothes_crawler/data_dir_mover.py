from os import name
from numpy.core.fromnumeric import mean
import glob
import shutil
from pathlib import Path
from collections import defaultdict

def shorten_name(name: str):
    return name[:4]

def shrink_dirs_mapping(dir: Path):
    # list 1st order dirs
    first_dirs = []
    dir_mapping = defaultdict(list)
    for p in dir.iterdir():
        if p.is_dir() and p.parent==dir:
            # print(p)
            dir_name = p.name
            first_dirs.append(p)
            shorten = shorten_name(dir_name)
            dir_mapping[shorten].append(p)
    
    print(f' Total dirs {len(first_dirs)}')
    print(f' Total new_dirs {len(dir_mapping)}')
    per_dir_stat_list = [len(v) for k,v in dir_mapping.items()]
    print(f'New dir stat: median {sorted(per_dir_stat_list)[int(len(per_dir_stat_list)/2)]}, mean {mean(per_dir_stat_list)}, max {max(per_dir_stat_list)}, min {min(per_dir_stat_list)}')
    return dir_mapping

def move_files(dir: Path, dir_mapping: dict):
    for new_dir_name, dir_list in dir_mapping.items():
        new_dir = Path(dir/ new_dir_name)
        print(f'Move {new_dir_name}:{len(dir_list)}')
        if not new_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
        for dir_ in dir_list:
            dest_dir = Path(new_dir/Path(dir_).name)
            if Path(dir_).exists() and not dest_dir.exists():
                shutil.move(dir_, str(dest_dir))




if __name__ == '__main__':
    dir = Path('/home/deeplab/datasets/custom_fashion/data/lamoda_ru')
    # dir_mapping = shrink_dirs_mapping(dir)
    dir = Path('/home/deeplab/datasets/custom_fashion/data/wildberries_ru')
    # dir = Path('/home/deeplab/datasets/custom_fashion/demo')
    dir_mapping = shrink_dirs_mapping(dir)
    # move_files(Path(dir.parent / (dir.name+'_')), dir_mapping)
