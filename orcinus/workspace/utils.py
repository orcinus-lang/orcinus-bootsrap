import os

from orcinus.exceptions import OrcinusError


def convert_module_name(filename, path):
    fullname = os.path.abspath(filename)
    if not fullname.startswith(path):
        raise OrcinusError(f"Not found file `{filename}` in package `{path}`")

    module_name = fullname[len(path):]
    module_name, _ = os.path.splitext(module_name)
    module_name = module_name.strip(os.path.sep)
    module_name = module_name.replace(os.path.sep, '.')
    return module_name


def convert_filename(module_name, path):
    filename = module_name.replace('.', os.path.sep) + '.orx'
    return os.path.join(path, filename)
