from ats.main import Cache


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
