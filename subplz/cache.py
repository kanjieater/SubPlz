from dataclasses import dataclass
from pathlib import Path


@dataclass(eq=True)
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool
    ask: bool
    overwrite: bool
    memcache: dict

    def get_name(self, filename, chid):
        return filename + '.' + str(chid) +  '.' + self.model_name + ".subs"

    def get(self, filename, chid):
        fn = self.get_name(filename, chid)
        if fn in self.memcache: return self.memcache[fn]
        if not self.enabled: return
        if (q := Path(self.cache_dir) / fn).exists():
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        # if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        fn =  self.get_name(filename, chid)
        p = cd / fn
        if p.exists():
            if self.ask:
                prompt = f"Cache for file {filename}, chapter id {chid} already exists. Overwrite?  [y/n/Y/N] (yes, no, yes/no and don't ask again) "
                while (k := input(prompt).strip()) not in ['y', 'n', 'Y', 'N']: pass
                self.ask = not (k == 'N' or k == 'Y')
                self.overwrite = k == 'Y' or k == 'y'
            if not self.overwrite: return content

        if 'text' in content:
            del content['text']
        if 'ori_dict' in content:
            del content['ori_dict']

        # Some of these may be useful but they just take so much space
        # for i in content['segments']:
        #     if 'words' in i:
        #         del i['words']
        #     del i['id']
        #     del i['tokens']
        #     del i['avg_logprob']
        #     del i['temperature']
        #     del i['seek']
        #     del i['compression_ratio']
        #     del i['no_speech_prob']

        self.memcache[fn] = content
        p.write_bytes(repr(content).encode('utf-8'))
        return content



def get_cache(backend, cache_inputs):
    overwrite_cache = cache_inputs.overwrite_cache
    enabled = cache_inputs.use_cache
    cache_dir = cache_inputs.cache_dir
    model_name = backend.model_name

    cache = Cache(
        model_name,
        enabled=enabled,
        cache_dir=cache_dir,
        ask=not overwrite_cache,
        overwrite=overwrite_cache,
        memcache={},
    )
    return cache

