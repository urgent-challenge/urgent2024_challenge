from pathlib import Path

from oct2py import octave


def setup_octave():
    matlab_script_dir = Path(__file__).parent.parent.resolve()
    matlab_script_dir = matlab_script_dir / 'PQevalAudio-v1r0/PQevalAudio'
    octave.addpath(str(matlab_script_dir))
    additional_dirs = ['Patt', 'Misc', 'MOV', 'CB']
    for adir in additional_dirs:
        adir_path = matlab_script_dir / adir
        octave.addpath(str(adir_path))
