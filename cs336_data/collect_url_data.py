import gzip
import random
import subprocess
import submitit
from pathlib import Path

def subsample_urls(wiki_urls_path, n_samples=10000):
    urls = []
    with gzip.open(wiki_urls_path, 'rt') as f:
        for line in f:
            url = line.strip()
            if url.startswith('http'):
                urls.append(url)

    return random.sample(urls, min(n_samples, len(urls)))

# what is actually run by submitit
def scrape_chunk(urls, chunk_id, output_dir):
    url_file = f'{output_dir}/urls_chunk_{chunk_id}.txt'
    with open(url_file, 'w') as f:
        for url in urls:
            f.write(url + '\n')

    warc_name = f'{output_dir}/chunk_{chunk_id}'
    cmd = [
        'wget',
        '--timeout=5',
        '-i', url_file,
        f'--warc-file={warc_name}',
        '-O', '/dev/null'
    ]

    subprocess.run(cmd)
    Path(url_file).unlink()

    return f'{warc_name}.warc.gz'

def main():
    wiki_urls = '/data/wiki/enwiki-20240420-extracted_urls.txt.gz'
    output_dir = '/data/c-aalag/sampled_url_data'
    Path(output_dir).mkdir(exist_ok=True)

    n_samples = 10000
    chunk_size = 100

    # print(f"{n_samples} urls")
    all_urls = subsample_urls(wiki_urls, n_samples)

    chunks = [all_urls[i : i + chunk_size] for i in range(0, len(all_urls), chunk_size)]
    # print(f"{len(chunks)} chunks of urls")

    executor = submitit.AutoExecutor(folder="/data/c-aalag/submitit_logs")
    executor.update_parameters(
        timeout_min=30,
        cpus_per_task=1,
        mem_gb=100,
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos"
    )

    jobs = []
    for i, chunk in enumerate(chunks):
        job = executor.submit(scrape_chunk, chunk, i, output_dir)
        jobs.append(job)

    # print(f"{len(jobs)} jobs")

    warc_files = []
    for job in jobs:
        try:
            result = job.result()
            warc_files.append(result)
            # print(f"Completed: {result}")
        except Exception as e:
            print(f"job failed: {e}")

    print(f"\n success in writing {len(warc_files)} WARC files to {output_dir}")

if __name__ == '__main__':
    main()