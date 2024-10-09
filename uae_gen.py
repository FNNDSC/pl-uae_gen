#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter

from chris_plugin import chris_plugin, PathMapper
import torch
import numpy as np
import mat73
from scipy.io import savemat, loadmat
from tqdm import tqdm
import os

__version__ = '1.0.0'

DISPLAY_TITLE = r"""
       _                                             
      | |                                            
 _ __ | |______ _   _  __ _  ___     __ _  ___ _ __  
| '_ \| |______| | | |/ _` |/ _ \   / _` |/ _ \ '_ \ 
| |_) | |      | |_| | (_| |  __/  | (_| |  __/ | | |
| .__/|_|       \__,_|\__,_|\___|   \__, |\___|_| |_|
| |                           ______ __/ |           
|_|                          |______|___/            
"""


parser = ArgumentParser(description='Find activation energy from multiple layers of VGGNet',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--pattern', default='*0.mat', type=str,
                    help='input file filter glob')
parser.add_argument('-V', '--version', action='version',
                    version=f'%(prog)s {__version__}')

# The main function of this *ChRIS* plugin is denoted by this ``@chris_plugin`` "decorator."
# Some metadata about the plugin is specified here. There is more metadata specified in setup.py.
#
# documentation: https://fnndsc.github.io/chris_plugin/chris_plugin.html#chris_plugin
@chris_plugin(
    parser=parser,
    title='pl-uae_gen',
    category='',                 # ref. https://chrisstore.co/plugins
    min_memory_limit='100Mi',    # supported units: Mi, Gi
    min_cpu_limit='1000m',       # millicores, e.g. "1000m" = 1 CPU core
    min_gpu_limit=0              # set min_gpu_limit=1 to enable GPU
)
def main(options: Namespace, inputdir: Path, outputdir: Path):
    """
    *ChRIS* plugins usually have two positional arguments: an **input directory** containing
    input files and an **output directory** where to write output files. Command-line arguments
    are passed to this main method implicitly when ``main()`` is called below without parameters.

    :param options: non-positional arguments parsed by the parser given to @chris_plugin
    :param inputdir: directory containing (read-only) input files
    :param outputdir: directory where to write output files
    """

    print(DISPLAY_TITLE)

    # Typically it's easier to think of programs as operating on individual files
    # rather than directories. The helper functions provided by a ``PathMapper``
    # object make it easy to discover input files and write to output files inside
    # the given paths.
    #
    # Refer to the documentation for more options, examples, and advanced uses e.g.
    # adding a progress bar and parallelism.
    mapper = PathMapper.file_mapper(inputdir, outputdir, glob=options.pattern)
    for input_file, output_file in mapper:
        def is_basename(x:str): return os.path.dirname(x) == ""

        def uae(f_map):
            return torch.sum(f_map)/f_map.numel()

        def hook_fn(module, input, output):
            result = uae(output[0])
            activation_outputs.append(result)

        activation_outputs = []

        patient_id = os.path.splitext(os.path.basename(input_file))[0]
        print(patient_id)
        patient_id = patient_id.split('_')[0]
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        # print(model)
        hooks = []
        activation_outputs = []
        layers = [2,4,7,9,12,14,16,19,21,23,26,28,30]
        for layer in layers:
            hook = model.features[layer].register_forward_hook(hook_fn)
            hooks.append(hook)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        model = model.to(device)

        mat = mat73.loadmat(input_file)
        mat = mat['tf_images_mat']
        
        epochs = mat.shape[0]
        channels = mat.shape[1]
        results = np.zeros((epochs,channels,len(layers)))
        activation_outputs = []

        for epoch in tqdm(range(epochs)):#epochs
            for channel in range(channels):#channels
                img = mat[epoch,channel,:,:,:]
                img = img.reshape(-1,img.shape[2],img.shape[0],img.shape[1])
                img = torch.tensor(img, dtype=torch.float32)
                img = img.to(device)
                res = model(img)
                np_activ = np.array([tensor.item() for tensor in activation_outputs])
                results[epoch,channel,:] = np_activ
                activation_outputs = []
        
        mdic = {"id": patient_id, "uae_vals": results}
        if is_basename(output_file):
            output_file = os.path.join(outputdir,output_file)
        savemat(output_file, mdic)
        print('Completed: ',patient_id, mat.shape)


if __name__ == '__main__':
    main()
