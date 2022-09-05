import sys
from PIL import Image
from pathlib import Path as P

args = sys.argv[1:]
args = list(map(P, args))
for f in args:
    img=Image.open(str(f))
    img.save(f'{f.with_suffix(".ico")}')

