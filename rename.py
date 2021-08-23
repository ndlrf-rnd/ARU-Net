import glob
import re
import os
for fp in glob.glob('./**/model*'):
    decomposed = re.findall("^(.+)[\\\/]([^\\\/\.0-9]+)([0-9]+)(\.[^.\\\/]+)*$", fp)
    if len(decomposed) > 0:
        folder, prefix, epoch, suffix = decomposed[0]
        epoch = int(epoch)
        output_path = os.path.join(
            folder,
            '{}-{:02d}{}'.format(prefix, epoch, suffix)
        )
        print(fp, '->', output_path, [folder, prefix, epoch, suffix])
        os.rename(fp, output_path)