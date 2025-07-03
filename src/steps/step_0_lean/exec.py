

import json
import shutil
import random
import networkx as nx
from copy import copy
from pathlib import Path
from loguru import logger
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Union

import lean_dojo
from lean_dojo import *
from lean_dojo.constants import LEAN4_PACKAGES_DIR

random.seed(3407)  # https://arxiv.org/abs/2109.08203

URL = "https://github.com/leanprover-community/mathlib4"
COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"
DST_DIR = Path("../leandojo_benchmark_4")
NUM_VAL = NUM_TEST = 2000



repo = LeanGitRepo(URL, COMMIT)
traced_repo = trace(repo)