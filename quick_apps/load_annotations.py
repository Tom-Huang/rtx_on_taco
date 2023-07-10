from collections import defaultdict
import numpy as np
from pathlib import Path
import hydra


@hydra.main(config_path="./config", config_name="load_annotations")
def main(cfg):
    root_data_dir = Path(cfg.root_data_dir)
    annotation_path = root_data_dir / cfg.lang_ann_folder / "auto_lang_ann.npy"
    data = np.load(annotation_path, allow_pickle=True)[()]
    print(data.keys())
    print("info: ", data["info"].keys())
    print("      episoded: ", data["info"]["episodes"][:20])
    print("      indx: ", data["info"]["indx"][:20])

    print("language: ", data["language"].keys())
    print("         task:", data["language"]["task"][:20])
    task2ids_dict = defaultdict(list)
    for id, task in enumerate(data["language"]["task"]):
        task2ids_dict[task].append(id)

    for k, v in task2ids_dict.items():
        print(k, len(v))

    task = "open_drawer"
    ids_list = task2ids_dict[task]
    for id in ids_list[:20]:
        print(data["info"]["indx"][id])
        print(data["language"]["ann"][id])


if __name__ == "__main__":
    main()
