from dataclasses import dataclass, field
from pathlib import Path


@dataclass(eq=True)
class Cache:
    model_name: str
    cache_dir: str
    enabled: bool
    memcache: dict = field(default_factory=dict)
    overwrite: bool = False
    cached_msg: bool = True # probably a better way to do this

    def get_name(self, filename, chid):
        return filename + "." + str(chid) + "." + self.model_name + ".subs"

    def get(self, filename, chid):
        if self.overwrite:
            self.overwrite = False
            self.cached_msg = False
            return
        fn = self.get_name(filename, chid)
        if fn in self.memcache:
            return self.memcache[fn]
        if not self.enabled:
            return
        if (q := Path(self.cache_dir) / fn).exists():
            if self.cached_msg:
                self.cached_msg = False
                print(f"💾 Cache hit for '{fn}' found on disk. If you want to regenerate the transcript for this file use the `--overwrite-cache` flag")
            return eval(q.read_bytes().decode("utf-8"))

    def put(self, filename, chid, content):
        # if not self.enabled: return content
        cd = Path(self.cache_dir)
        cd.mkdir(parents=True, exist_ok=True)
        fn = self.get_name(filename, chid)
        p = cd / fn
        if p.exists():
            if not self.overwrite:
                return content

        if "text" in content:
            del content["text"]
        if "ori_dict" in content:
            del content["ori_dict"]

        self.memcache[fn] = content
        p.write_bytes(repr(content).encode("utf-8"))
        return content


def get_cache(cache_inputs):
    overwrite_cache = cache_inputs.overwrite_cache
    enabled = cache_inputs.use_cache
    cache_dir = cache_inputs.cache_dir

    cache = Cache(
        cache_inputs.model_name,
        enabled=enabled,
        cache_dir=cache_dir,
        overwrite=overwrite_cache,
        memcache={},
    )
    return cache
