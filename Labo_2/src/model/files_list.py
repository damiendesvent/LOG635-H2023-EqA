from glob import glob

def files_list(root):
    return glob(root + '/**/*.jpg', recursive=True)