import json  # for exec.ipynb
import os
import subprocess
import sys

from midi2audio import FluidSynth
from pydub import AudioSegment
from pathlib import Path

GENRES_LIST = ( # noqa WPS317
	'pop', 'jazz', 'rock', 'blues', 'classical', 'country',
	'soul', 'rap', 'latin', 'folk', 'electro', '[UNK]',
)


def main(args):  # noqa: WPS210
	"""Process input parameters and starts generation.
	Args:
		args: input arguments
	"""
	model_weights = args[1]
	time_ms = args[2]
	sentiment = args[3]
	genre = args[4]
	# output_dir = '/data/generation/' + args[5]
	output_dir = Path('/storage/arkady/Glinka/music-transformer/video_conditioning/') / str(args[5])
	output_dir = str(output_dir)
	output_name = args[6]
	print('=======', output_dir, output_name)

	with open('generate_sample_parameters.json', 'r') as parameters_file: # loading base parameters
		parameters = json.load(parameters_file)  # noqa: WPS110

	parameters['saving_options']['output_dir'] = output_dir
	parameters['saving_options']['output_folder_name'] = output_name
	parameters['generation_params']['additional_features_params']['genre'] = GENRES_LIST[int(genre)]
	#parameters['generation_params']['additional_features_params']['sentiment_per_token'] = int(sentiment)
	parameters['generation_params']['additional_features_params']['timing'] = int(int(time_ms))
	parameters['model_weights'] = model_weights

	# kvm nnotes:
	parameters['generation_params']['additional_features_params']['nnotes']['preset'] = \
		[
			{
				"borders": [64 * i, 64 * (i + 1)],
				"value": 1 + i * 9
			}
			for i in range(2)
		]

	if len(args) > 7:
		parameters['random_seed'] = int(args[7])

	os.makedirs(output_dir, exist_ok=True)
	out_parameters_file_name = '{0}/generate_sample_{1}_parameters.json'.format(output_dir, output_name)
	with open(out_parameters_file_name, 'w') as out_parameters_file: # Saving custom parameters. Generated sample would use them instead of basic ones
		json.dump(parameters, out_parameters_file, indent=4)

	subprocess.run(
		['/storage/arkady/miniconda3/envs/transformer_production/bin/python',
		 '/storage/arkady/Glinka/music-transformer/video_conditioning/code3/generate_sample.py', 
		 output_dir,
		 output_name
		],
	)

	# wav_file = '{0}/{1}.wav'.format(output_dir, output_name)

	# FluidSynth(sound_font='default_sound_font.sf2').midi_to_audio('{0}/{1}/{1}.midi'.format(output_dir, output_name), wav_file)
	# AudioSegment.from_wav(wav_file).export('{0}/{1}.mp3'.format(output_dir, output_name), format='mp3')


if __name__ == '__main__':
	main(sys.argv)
