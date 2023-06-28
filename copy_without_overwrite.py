# It copies a file if it does not exist at the destination path.

import os
import sys
import shutil


def main(argv):
    if len(argv) != 2:
        raise Exception(f'Error: The program needs two arguments, but you gave {len(argv)} arguments!')

    inp = argv[0]
    out = argv[1]
    try:
        if not os.path.exists(out):
            shutil.copyfile(inp, out)
        else:
            print(f"We did not copy \"{inp}\" because the file \"{out}\" exists at destination!")
    except Exception:
        raise Exception(f"Error occurred while copying \"{inp}\" file to \"{out}\"!")


if __name__ == '__main__':
    main(sys.argv[1:])
