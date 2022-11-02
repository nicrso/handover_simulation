import os

def getDataPath(type):
    resdir = os.path.join(os.path.dirname(__file__), type)
    return resdir

def getURDFPath(name, type):
    urdf_path = os.path.join(getDataPath(type),name + '.urdf')
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)
    return urdf_path
