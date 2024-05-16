from ats.main import Cache

def get_cache(args, model):
    model_name = args.get("model")
    overwrite, overwrite_cache = args.pop('overwrite'), args.pop('overwrite_cache')
    cache = Cache(model_name=model_name, enabled=args.pop("use_cache"), cache_dir=args.pop("cache_dir"),
                  ask=not overwrite_cache, overwrite=overwrite_cache,
                  memcache={})
    return cache