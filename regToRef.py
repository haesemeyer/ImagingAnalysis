import subprocess as sp
from mh_2P import UiGetFile


def name(filename, ext):
    # remove extension
    end = filename.lower().find(ext)
    if end == -1:
        raise ValueError(filename + "Is not an " + ext + " file")
    return filename[:end]


def affine_name(filename):
    return name(filename, ".nrrd") + "_affine.xform"


def warp_name(filename):
    return name(filename, ".nrrd") + "_ffd5.xform"


def out_name(filename):
    return name(filename, ".nrrd") + "_warped.nrrd"


def make_output_command(name):
    return '-o ' + name


def make_floating_command(name):
    return '--floating ' + name


def make_initial_command(name):
    return '--initial ' + name


def crop_heuristics(filename):
    # tries to extract the appropriate reference crop based on the filename
    if "FB" in filename:
        return '--crop-index-ref "0, 935, 0, 770, 1380, 100"'
    elif "MidHB" in filename:
        return '--crop-index-ref "0, 0, 0, 770, 975, 100"'
    else:
        print("Could not identify brain region from filename. No cropping!")
        return ''


def assemble_commands(com_list):
    return ' '.join(com_list)


if __name__ == "__main__":
    ref_brain_path = "E:/Dropbox/ReferenceBrainCreation/H2BGc6s_Reference_8.nrrd"
    registration_commands_base = ['registration', '--dofs 6,9', '--verbose', '--exploration 30', '--accuracy 0.8',
                                  '--auto-multi-levels 4', '--initxlate', '--threads 6']
    warp_commands_base = ['warp', '--grid-spacing 80', '--refine 3', '--jacobian-weight 5e-5', '--exploration 26',
                          '--energy-weight 1e-1', '--coarsest 8', '--match-histograms', '--accuracy 0.8', '--verbose',
                          '--threads 6']
    reformat_commands_base = ['reformatx']
    float_names = UiGetFile([('Nrrd stack', '.nrrd')], multiple=True)
    for fn in float_names:
        an = affine_name(fn)
        wn = warp_name(fn)
        on = out_name(fn)
        crop_command = crop_heuristics(fn)
        reg_command = assemble_commands(registration_commands_base + [crop_command, make_output_command(an),
                                                                      ref_brain_path, fn])
        print(reg_command, flush=True)
        print("", flush=True)
        sp.run(reg_command)
        warp_command = assemble_commands(warp_commands_base + [crop_command, make_output_command(wn),
                                                               make_initial_command(an), ref_brain_path, fn])
        sp.run(warp_command)
        print("", flush=True)
        print(warp_command, flush=True)
        print("", flush=True)
        reform_command = assemble_commands(reformat_commands_base + ["--outfile " + on,
                                                                     make_floating_command(fn), ref_brain_path, wn])
        print("", flush=True)
        print(reform_command, flush=True)
        print("", flush=True)
        sp.run(reform_command)
