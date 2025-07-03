import os
from pathlib import Path

projects_path = Path("/home/qyh/DATASET/BugsInPy/projects/")
for bug_info_file in list(projects_path.rglob("bug.info"))[:1]:
    proj = bug_info_file.parents[2].name
    bug_id = bug_info_file.parent.name

    os.system(
        f"bugsinpy-checkout -p {proj} -v 0 -i {bug_id} -w /home/qyh/projects/TreeFL/dataset/buggy_codebase/{proj}/{bug_id}"
    )
    os.chdir(
        f"/home/qyh/projects/TreeFL/dataset/buggy_codebase/{proj}/{bug_id}/{proj}"
    )
    os.system("bugsinpy-compile")
