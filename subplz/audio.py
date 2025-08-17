from subplz.utils import get_iso639_2_lang_code

def score_audio_stream(stream_info: dict, target_iso_lang: str | None) -> int:
    """Assigns a preference score to an audio stream."""
    score = 0
    tags = stream_info.get("tags", {})
    title = tags.get("title", "").lower()
    stream_lang = stream_info.get("tags", {}).get("language", "").lower()

    # 1. Language Match (highest importance)
    if target_iso_lang and stream_lang == target_iso_lang:
        score += 100
    elif target_iso_lang and target_iso_lang in title: # Less reliable but a fallback
        score += 20


    # 2. Avoid undesirable tracks (negative scores)
    commentary_keywords = ["commentary", "comment", "comms", "director"]
    ad_keywords = ["audio description", "descriptive", "dvs", " ad ", " visually impaired", " vi "]
    isolated_keywords = ["instrumental", "music only", "effects only", "sfx", "m&e"]

    if any(kw in title for kw in commentary_keywords):
        score -= 200
    if any(kw in title for kw in ad_keywords):
        score -= 200
    if any(kw in title for kw in isolated_keywords):
        score -= 150

    # 3. Prefer Stereo for transcription (can be adjusted)
    channels = stream_info.get("channels", 0)
    channel_layout = stream_info.get("channel_layout", "").lower()
    if channels == 2 or "stereo" in channel_layout or "stereo" in title:
        score += 10
    elif channels > 2: # Multi-channel might be fine, but stereo is often simpler
        score += 5 # Slight preference for having more channels over unknown

    # 4. Prefer default track if language is unknown or multiple matches
    if stream_info.get("disposition", {}).get("default", 0):
        score += 5

    # 5. Prefer non-dubbed if language matches or is primary
    if stream_info.get("disposition", {}).get("dub", 0):
        score -= 2 # Slight penalty for dubs if other factors are equal

    # 6. Codec quality (very rough preference)
    codec_name = stream_info.get("codec_name", "").lower()
    if codec_name in ["flac", "pcm_s16le", "truehd", "dts-hd_ma"]:
        score += 3 # Lossless
    elif codec_name in ["aac", "opus", "vorbis", "dts", "eac3", "ac3"]:
        score += 1 # Good lossy
    # mp3 might get 0 or negative if others are available

    return score


def get_audio_idx(all_streams: list, target_lang_code: str, path: str) -> dict | None:
    """
    Finds the best matching audio stream based on language and other heuristics.
    Returns the stream's info dictionary or None.
    """
    audio_streams = [s for s in all_streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        print("❗No audio streams found in the media.")
        return None

    standardized_target_lang = get_iso639_2_lang_code(target_lang_code)
    if not standardized_target_lang:
        print(f"🦈Could not standardize input language code for the audio stream '{target_lang_code}'. Will select based on other heuristics or first available.")

    scored_streams = []
    for stream in audio_streams:
        score = score_audio_stream(stream, standardized_target_lang)
        scored_streams.append({"score": score, "stream_info": stream})

    # Sort streams by score in descending order
    scored_streams.sort(key=lambda x: x["score"], reverse=True)

    if not scored_streams: # Should not happen if audio_streams is not empty
        return None

    best_match = scored_streams[0]

    # Logging for transparency
    if best_match["score"] < 0: # All tracks were undesirable
        print(f"🦈All audio tracks scored negatively. Best undesirable match selected (score: {best_match['score']}) for file: {path}")
    elif standardized_target_lang and best_match["stream_info"].get("tags",{}).get("language","").lower() != standardized_target_lang:
        print(f"🦈No direct language match for the audio stream '{target_lang_code}' (standardized: '{standardized_target_lang}') for file: {path}")
        # print(f"Selected best alternative (score: {best_match['score']}): Stream index {best_match['stream_info'].get('index')}, Lang='{best_match['stream_info'].get('tags',{}).get('language','N/A')}'")
    elif not standardized_target_lang and target_lang_code: # Added 'and target_lang_code' here as well for consistency
         print(f"🦈Target language for the audio stream '{target_lang_code}' was not recognized. Selected best available stream based on other heuristics for file: {path}")

    print(f"🚣 Selected stream (Index: {best_match['stream_info'].get('index')}, Score: {best_match['score']}, Lang: {best_match['stream_info'].get('tags',{}).get('language','N/A')}, Title: {best_match['stream_info'].get('tags',{}).get('title','N/A')}) for file: {path}")
    return best_match["stream_info"]