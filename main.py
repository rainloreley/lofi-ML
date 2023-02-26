import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy
import shutil
from music21 import converter, instrument, note, stream, chord
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def initTraining(epochs, batch_size):

    print("==================")
    print(f"{bcolors.HEADER}Preparing model training with following config:{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Epochs:{bcolors.ENDC} {epochs}")
    print(f"{bcolors.BOLD}Batch Size:{bcolors.ENDC} {batch_size}")
    print("==================")

    time.sleep(3)

    # get all notes from the midi training data
    (notes, n_vocab) = load_midi_notes()

    # prepare sequences for use
    (network_input, normalized_input, network_output, pitchnames) = prepare_sequences(notes, n_vocab)

    # create model
    model = create_model(normalized_input, n_vocab)

    # train model
    iso_date = train(model, normalized_input, network_output, epochs, batch_size)

    # find best model from model checkpoints
    find_best_model(iso_date)

    print("==================")
    print(f"{bcolors.OKGREEN}Done!{bcolors.ENDC}")

def initGenerating(weight_name, notes_amount):
    print("==================")
    print(f"{bcolors.HEADER}Preparing music generation with following config:{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Weight file:{bcolors.ENDC} {weight_name}")
    print(f"{bcolors.BOLD}Notes amount:{bcolors.ENDC} {notes_amount}")
    print("==================")


    # get all notes from the midi training data
    (notes, n_vocab) = load_midi_notes()

    # prepare sequences for use
    (network_input, normalized_input, network_output, pitchnames) = prepare_sequences(notes, n_vocab)

    # create model
    model = create_model(normalized_input, n_vocab)
    model.load_weights(weight_name)

    prediction_output = generate_music(model, network_input, n_vocab, pitchnames, notes_amount)

    build_midi_output(prediction_output)
    print("==================")
    print(f"{bcolors.OKGREEN}Done!{bcolors.ENDC}")

def load_midi_notes():
    notes = []

    for file in glob.glob("trainingmidis/*.mid"):
        midi = converter.parse(file)
        notes_to_parse = None

        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    n_vocab = len(set(notes))
    return notes, n_vocab

def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)

    return network_input, normalized_input, network_output, pitchnames

def create_model(normalized_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(normalized_input.shape[1], normalized_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, normalized_input, network_output, epochs, batch_size):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(physical_devices[0], device_type='GPU')

    iso_date = datetime.now().isoformat()
    checkpoints_folder = f"modelcheckpoints/{iso_date}"

    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)


    filepath = checkpoints_folder + "/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(normalized_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    return iso_date

def find_best_model(iso_date):
    best_loss: float = 100
    best_epoch = None
    best_file = None

    for file in glob.glob(f"modelcheckpoints/{iso_date}/*.hdf5"):
        filename = file.split("/")[len(file.split("/")) - 1]
        splitted_filename = filename.split("-")
        loss_value = float(splitted_filename[3])
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = splitted_filename[2]
            best_file = file
    shutil.copyfile(best_file, f"bestmodels/weights__{iso_date}.hdf5")
    print("==================")
    print(f"{bcolors.OKCYAN} Saved best model (Epoch {best_epoch}; loss: {best_loss}) to {bcolors.HEADER}bestmodels/weights__{iso_date}.hdf5{bcolors.ENDC}")

def generate_music(model, network_input, n_vocab, pitchnames, notes_amount):
    # generate music using model
    start = numpy.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = network_input[start]
    prediction_output = []
    for note_index in range(notes_amount):
        print(f"{bcolors.HEADER}Generating note {note_index + 1} of {notes_amount}{bcolors.ENDC}")
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=1)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def build_midi_output(prediction_output):
    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    iso_date = datetime.now().isoformat()

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'output/test_output-{iso_date}.mid')

    print("==================")
    print(f"{bcolors.OKCYAN}Saved MIDI to {bcolors.HEADER}'output/test_output-{iso_date}.mid'{bcolors.ENDC}")

def get_option_value(opts, name, default):
    if any(f"--{name}=" in opt for opt in opts):
        full_option = [opt for opt in opts if f"--{name}=" in opt][0]
        splitted_argument = full_option.split("=")
        if len(splitted_argument[1]) == 0:
            print(f"{bcolors.FAIL}Invalid argument ('{full_option}'), exiting...{bcolors.ENDC}")
            exit(1)
        else:
            return splitted_argument[1]
    else:
        return default

def check_if_folders_exist():
    bestmodels_folder = "bestmodels"
    modelcheckpoints_folder = "modelcheckpoints"
    output_folder = "output"
    trainingmidis_folder = "trainingmidis"
    if not os.path.exists(bestmodels_folder):
        os.makedirs(bestmodels_folder)
    if not os.path.exists(modelcheckpoints_folder):
        os.makedirs(modelcheckpoints_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(trainingmidis_folder):
        print(f"{bcolors.FAIL}Folder 'trainingmidis' does not exist, no MIDIS available! Exiting...{bcolors.ENDC}")
        exit(1)

if __name__ == "__main__":
    check_if_folders_exist()
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if len(args) == 0:
        print(f"{bcolors.FAIL}No arguments parsed, exiting...{bcolors.ENDC}")
        exit(1)
    command = args[0]
    if command == "train":
        epochs = int(get_option_value(opts, "epochs", 500))
        batch_size = int(get_option_value(opts, "batchsize", 164))
        initTraining(epochs, batch_size)
    elif command == "gen":
        notes_amount = int(get_option_value(opts, "notes", 100))
        weight_name = get_option_value(opts, "weightfile", "")
        if weight_name == "":
            print(f"{bcolors.FAIL}No weight name given! Exiting...{bcolors.ENDC}")
            exit(1)
        elif not os.path.isfile(weight_name):
            print(f"{bcolors.FAIL}Weight file '{weight_name}' does not exist! Exiting...{bcolors.ENDC}")
            exit(1)
        initGenerating(weight_name, notes_amount)
    else:
        print(f"{bcolors.FAIL}Unknown command ('{command}'), exiting...{bcolors.ENDC}")
        exit(1)

