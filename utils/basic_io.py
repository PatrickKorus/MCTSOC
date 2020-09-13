import os
import errno
import pickle

""" Basic container input/output using Pickle """


def dump(target, data):
    """
    Dumps container at target path
    :param target: path, where data should be dumped
    :param data: the data to be dumped
    """
    dir_name = os.path.dirname(target)
    if not os.path.exists(dir_name) and dir_name != '':
        try:
            os.makedirs(dir_name)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(target, "wb+") as sourceFile:
        pickle.dump(data, sourceFile)


def load(source):
    """
    Load a dump created with "dump(target, data)"
    :param source: the path of the dump to be loaded
    """
    try:
        with open(source, "rb") as sourceFile:
            dump = pickle.load(sourceFile)
            return dump
    except FileNotFoundError:
        return None


def list_dumps(source, file_type=".dump"):
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
