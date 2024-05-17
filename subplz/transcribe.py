import time
import concurrent.futures as futures


def transcribe(streams, model, cache, be):
	max_workers = be.threads
	print('Transcribing...')
	# copy the args from
	args = {
		'language': be.language,
		'initial_prompt': be.initial_prompt,
		'length_penalty': be.length_penalty,
		'temperature': be.temperature,
		'beam_size': be.beam_size,
		'patience': be.patience,
		'suppress_tokens': be.suppress_tokens,
		'prepend_punctuations': be.prepend_punctuations,
		'append_punctuations': be.append_punctuations,
		'compression_ratio_threshold': be.compression_ratio_threshold,
		'log_prob_threshold': be.log_prob_threshold,
		'condition_on_previous_text': be.condition_on_previous_text,
		'no_speech_threshold': be.no_speech_threshold,
		'word_timestamps': be.word_timestamps,
	}
	# TODO: not faster-whisper
	# logprob_threshold

	s = time.monotonic()
	with futures.ThreadPoolExecutor(max_workers) as p:
		r = []
		for i in range(len(streams)):
			for j, v in enumerate(streams[i][2]):
				# TODO add **args back in
				future = p.submit(lambda x: x.transcribe(model, cache, **args), v)
				r.append(future)
		futures.wait(r)

		for future in r:
			# try:
			future.result()  # Get the result to check for exceptions
			# except Exception as e:
			#     print("An error occurred while transcribing", e)

	print(f'Transcribing took: {time.monotonic()-s:.2f}s')
	return streams


# def transcribe(streams, model, cache, temperature, threads, args):
#     for i in range(len(streams)):
#         for j, v in enumerate(streams[i][2]):
#             v.transcribe(model, cache, temperature=temperature, **args)
#     # print('Transcribing...')
#     # s = time.monotonic()
#     # with futures.ThreadPoolExecutor(max_workers=threads) as p:
#     #     r = []
#     #     for i in range(len(streams)):
#     #         for j, v in enumerate(streams[i][2]):
#     #             r.append(p.submit(lambda x: x.transcribe(model, cache, temperature=temperature, **args), v))
#     #     futures.wait(r)
#     # print(f"Transcribing took: {time.monotonic()-s:.2f}s")
#     return streams
