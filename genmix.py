import csv
import glob
import random
import numpy as np
import scipy as sp
import scipy.signal
import fluidsynth
import pretty_midi
import tensorflow as tf
import sympy
seed = 12345


FP_SF2_PATH = "/usr/share/soundfonts/freepats-general-midi.sf2"
vilulia = "./shapenote_midi/312b.mid"
with open('freepats_instruments.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    insts = list(reader)
    freepats_inst_dict = {int(d['num']) - 1: d['instrument'] for d in insts}


midi_data = pretty_midi.PrettyMIDI(vilulia)
samples = midi_data.fluidsynth(sf2_path=FP_SF2_PATH)


def randomize_instruments(midi, inst_dict):
    """
    This function takes a midi object, a dictionary of avaiable instruments,
    and scrambles the instruments assigned to the midi tracks. the midi file is
    passed in by reference, so we only return the chosen instruments, not the
    midi file
    """
    instruments = []
    for inst in midi.instruments:
        # casting a dict to a list returns a list of only the keys
        inst.program = random.choice(list(inst_dict))
        instruments.append(inst.program)
    return np.array(instruments)


def get_current_insts(midi):
    """returns array containing current instrument assignments"""
    instruments = []
    for inst in midi.instruments:
        instruments.append(inst.program)
    return np.array(instruments)


def generate_mixes(midi, inst_dict, sf2_path, n=1000):
    """
    This function takes a midi object, a dictionary of avaialbe instruments,
    a path to a soundfont, and a number of mixes to generate. It returns a numpy
    array of mixes, and a second array of instrument numbers for those mixes.
    """

    num_insts = len(midi.instruments)
    inst_ary = np.random.choice(list(inst_dict), (n, num_insts))
    inst_ary[0] = get_current_insts(midi)

    orig_samples = midi.fluidsynth(sf2_path=sf2_path)
    samples_ary = np.zeros((n, len(orig_samples)))
    samples_ary[0] = orig_samples

    for i in range(1, n):
        inst_ary[i] = randomize_instruments(midi, inst_dict)
        samples_ary[i] = midi.fluidsynth(sf2_path=sf2_path)

    return samples_ary, inst_ary

def generate_mix(dataset_entry, sf2_path=FP_SF2_PATH):
    """
    This is the heavy lifting inner function that takes a midi object as
    well as a list of midi instrument numbers, then synthesizes it,
    computes the STFT and returns it.
    """
    filename, insts = dataset_entry
    midi = pretty_midi.PrettyMIDI(filename)

    num_insts = len(midi.instruments)
    # sanity check: the number of new instruments should match the current
    # number of instruments
    assert num_insts == len(insts)
    for i, inst in enumerate(midi.instruments):
        # assign the chosen instruments in the midi object
        midi.instruments[i].program = insts[i]
    # synthesize the midi file
    samples = midi.fluidsynth(sf2_path=sf2_path)
    # and return a spectrogram
    return sp.signal.stft(samples, fs=44100)

def generate_training_set(midi_directory="./shapenote_midi/",
        sf2_path=FP_SF2_PATH, insts=freepats_inst_dict, n=1000):
    """
    This function basically takes a directory of midi files, calls
    generate_mixes() on all of them, and assembles the results into something
    nice to be sent off to the neural net for training
    """

    files = glob.glob(midi_directory + "**.mid")
    files.sort()
    # Initiate a rng instance with a seed for redproducibility
    # 4 is chosen by a fair dice roll, guaranteed to be random :p
    # https://xkcd.com/221/
    rng = np.random.default_rng(4)
    n_files = len(files)
    # variations are expensive to compute so make a dictionary for them
    variations = {}
    # list where we will collate our results
    training_objects = []
    for i, file in enumerate(files):
        print(str(i) + " out of " + str(n_files)) 
        current_midi = pretty_midi.PrettyMIDI(file)
        n_insts = len(current_midi.instruments)
        # first check if we have already computed the variations for this
        # combination of instrument list and instrument number
        if n_insts not in variations:
            # This function makes a list of all possible unique variations of
            # n_insts instruments selected from insts, the avaialble instruments 
            # of the current sound font.
            var_gen = sympy.utilities.iterables.variations(insts.keys(),
                                                           n_insts)
            variations[n_insts] = np.array(list(var_gen))

        # choose n of those variations randomly (but reproducably)
        my_variations = rng.choice(variations[n_insts], size=n, replace=False)
        # my_variations = variations[n_insts][:n]
        
        training_objects += [[file, i] for i in my_variations]

    # Now the magic begins! cast this to a TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices(training_objects)
    dataset = dataset.map(generate_mix, num_parallel_calls=tf.data.AUTOTUNE, 
                deterministic=False)
    dataset = dataset.batch(10)
    dataset = dataset.prefetch(5)
    return dataset

    





 
