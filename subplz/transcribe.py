import time
import concurrent.futures as futures


def transcribe(streams, model, cache, temperature, threads, args):
    print('Transcribing...')
    s = time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=threads) as p:
        r = []
        for i in range(len(streams)):
            for j, v in enumerate(streams[i][2]):
                #TODO add **args back in
                future = p.submit(lambda x: x.transcribe(model, cache, temperature=temperature), v)
                r.append(future)
        futures.wait(r)

        for future in r:
            # try:
            future.result()  # Get the result to check for exceptions
            # except Exception as e:
            #     print("An error occurred while transcribing", e)


    print(f"Transcribing took: {time.monotonic()-s:.2f}s")
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