import sys
import time

try:
    from tap import Tap
except ImportError as e:
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    print(f'`>>>>>>>> from tap import Tap` failed, please run:      pip3 install typed-argument-parser     <<<<<<<<', file=sys.stderr, flush=True)
    time.sleep(5)
    raise e