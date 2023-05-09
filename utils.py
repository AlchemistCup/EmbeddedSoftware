from pathlib import Path

def find_starting_id(dir: Path):
    starting_id = 0
    for file in dir.iterdir():
        curr_id = int(file.stem.split('_')[-1])
        starting_id = max(starting_id, curr_id)
    return starting_id + 1