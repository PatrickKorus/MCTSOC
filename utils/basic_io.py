import os
import errno
import pickle

""" Basic container input/output using Pickle """


def ensure_path(target):
    """
    Makes sure path exists of target exists, while guarding against race condition.

    :param target: Directory as string e.g. 'path/to/dir/file.csv'
    """
    dir_name = os.path.dirname(target)
    if not os.path.exists(dir_name) and dir_name != '':
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def dump(target, data):
    """
    Dumps container at target path

    :param target: path, where data should be dumped
    :param data: the data to be dumped
    """
    ensure_path(target)
    with open(target, "wb+") as sourceFile:
        pickle.dump(data, sourceFile)
        sourceFile.close()


def write_string(target, text):
    """
    Writes text to target file.
    """
    ensure_path(target)
    with open(target, 'wt') as text_file:
        text_file.write(text)
        text_file.close()


def load(source):
    """
    Load a dump created with "dump(target, data)"

    :param source: the path of the dump to be loaded
    """
    try:
        with open(source, "rb") as sourceFile:
            dump = pickle.load(sourceFile)
            sourceFile.close()
            return dump
    except FileNotFoundError:
        return None


def list_files(source, file_type=".dump"):
    """
    Shows all dumps in a directory, with ending of file_type

    :param source: directory path
    :param file_type: ".dump" if not specified
    :return:
    """
    if not os.path.exists(source):
        return []
    dumps = []
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    for file in files:
        if file.endswith(file_type):
            dumps.append(file)
    return dumps
