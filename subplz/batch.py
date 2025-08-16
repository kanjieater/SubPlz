from pathlib import Path
from types import SimpleNamespace
from subplz.helpers import extract, rename, copy
from subplz.cli import RenameParams, ExtractParams, CopyParams, SyncData, GenData, SyncParams, GenParams, BatchParams, get_inputs
from subplz.sync import run_sync
from subplz.gen import run_gen


def run_batch(inputs):
    for directory in inputs.dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            print(f"❗ Skipping invalid directory: {dir_path}")
            continue

        print(f"\n--- Processing directory: {dir_path} ---")

        # Rename ja -> ab
        print("\n--- Renaming 'ja' subs to 'ab' for processing ---")
        rename_ja_args = ['rename', '-d', str(dir_path), '--lang-ext', 'ab', '--lang-ext-original', 'ja', '--unique']
        rename(get_inputs(rename_ja_args))

        # Extract ja -> tl
        print("\n--- Extracting & Verifying Native Target Language ('ja' -> 'tl') ---")
        extract_args = ['extract', '-d', str(dir_path), '--lang-ext', 'tl', '--lang-ext-original', 'ja', '--verify']
        extract(get_inputs(extract_args))

        # Alass Sync en -> as
        print("\n--- Alass Syncing ('en' + 'ab' -> 'as') ---")
        alass_args = [
            'sync', '-d', str(dir_path), '--lang-ext', 'as',
            '--lang-ext-original', 'en', '--lang-ext-incorrect', 'ab', '--alass'
        ]
        run_sync(get_inputs(alass_args))

        # SubPlz Sync ab -> ak
        print("\n--- SubPlz Syncing ('ab' -> 'ak') ---")
        subplz_sync_args = [
            'sync', '-d', str(dir_path), '--lang-ext', 'ak',
            '--lang-ext-original', 'ab', '--model', 'large-v3'
        ]
        run_sync(get_inputs(subplz_sync_args))

        # Gen az
        print("\n--- Generating subs ('az') ---")
        gen_args = ['gen', '-d', str(dir_path), '--lang-ext', 'az', '--model', 'large-v3']
        run_gen(get_inputs(gen_args))

        # Alass AI Sync az -> aa
        print("\n--- Alass AI Syncing ('az' + 'ab' -> 'aa') ---")
        alass_ai_args = [
            'sync', '-d', str(dir_path), '--lang-ext', 'aa',
            '--lang-ext-original', 'az', '--lang-ext-incorrect', 'ab', '--alass'
        ]
        run_sync(get_inputs(alass_ai_args))

        # Copy prioritized -> ja
        print("\n--- Copying best subtitle to 'ja' ---")
        copy_args = [
            'copy', '-d', str(dir_path), '--lang-ext', 'ja',
            '--lang-ext-priority', "tl", "as", "aa", "ak", "az", "ab",
            '--overwrite'
        ]
        copy(get_inputs(copy_args))

        print(f"\n--- All operations completed for: {dir_path} ---")
