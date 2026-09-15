import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collections import Counter
from src.data.voc import parse_voc_xml

ann = ROOT / 'data/rdd/Czech/train/annotations/xmls'
img = str(ROOT / 'data/rdd/Czech/train/images')
stats, label_counts, n = Counter(), Counter(), 0
for xml in sorted(ann.glob('*.xml')):
    s = parse_voc_xml(str(xml), img, stats)
    label_counts.update(s.labels)
    n += 1
print(f'parsed {n} files')
print('labels:', dict(label_counts))
print('corruption stats:', dict(stats))