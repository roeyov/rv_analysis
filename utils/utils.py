import os
import sys
import shutil


def make_runlog(output_dir, files_to_copy=None):
    os.makedirs(os.path.join(output_dir, 'runlog'), exist_ok=True)
    command_line_string = ' '.join(sys.argv)
    with open(os.path.join(output_dir, 'runlog','howto.txt'), 'w') as howto_file:
        howto_file.write(command_line_string)
    if files_to_copy is None:
        return
    for param_file in files_to_copy:
        shutil.copy(param_file, os.path.join(output_dir, 'runlog', os.path.basename(param_file)))

