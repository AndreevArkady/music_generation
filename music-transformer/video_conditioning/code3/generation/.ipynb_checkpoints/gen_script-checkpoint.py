import json
import os
import subprocess
import sys

from midi2audio import FluidSynth
from pydub import AudioSegment

from utilities.constants import GENRES_LIST


def main(args):  # noqa: WPS210
    """Process input parameters and starts generation.

    Args:
        args: input arguments
    """
    model_weights = args[1]
    seq_len = args[2]
    sentiment = args[3]
    genre = args[4]
    output_dir = args[5]
    output_name = args[6]

    with open('generation/generate_sample_parameters.json', 'r') as parameters_file: # loading base parameters
        parameters = json.load(parameters_file)  # noqa: WPS110

    parameters['output_dir'] = output_dir
    parameters['output_name'] = output_name
    parameters['generation_params']['additional_features_params']['genre'] = GENRES_LIST[int(genre)]
    parameters['generation_params']['additional_features_params']['sentiment_per_token'] = int(sentiment)
    parameters['generation_params']['n_generate'] = int(seq_len)
    parameters['model_weights'] = model_weights

    if len(args) > 7:
        parameters['random_seed'] = int(args[7])

    os.makedirs(output_dir, exist_ok=True)
    out_parameters_file_name = '{0}/generate_sample_{1}_parameters.json'.format(output_dir, output_name)
    with open(out_parameters_file_name, 'w') as out_parameters_file: # Saving custom parameters. Generated sample would use them instead of basic ones
        json.dump(parameters, out_parameters_file, indent=4)

    subprocess.run(
        [sys.executable, '-m', 'generation.generate_sample', output_dir, output_name],
    )

    wav_file = '{0}/{1}.wav'.format(output_dir, output_name)

    FluidSynth().midi_to_audio('{0}/{1}/{1}.midi'.format(output_dir, output_name), wav_file)
    AudioSegment.from_wav(wav_file).export('{0}/{1}.mp3'.format(output_dir, output_name), format='mp3')


if __name__ == '__main__':
    main(sys.argv)
