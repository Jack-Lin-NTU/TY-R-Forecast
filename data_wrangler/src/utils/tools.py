import os

def checkpath(path):
    if os.path.exists(path):
        return True
    else:
        return False
        
def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def make_path(path, workfolder=None):
    if path[0] == '~':
        p = os.path.expanduser(path)
    else:
        p = path
    
    if workfolder is not None and workfolder[0] == '~':
        workfolder = os.path.expanduser(workfolder)
    else:
        workfolder = workfolder
    
    if workfolder is not None and not os.path.isabs(p):
        return os.path.join(os.path.expanduser(workfolder), p)
    else:
        return p

def print_dict(d):
    for key, value in d.items():
        print('{}: {}'.format(key, value))