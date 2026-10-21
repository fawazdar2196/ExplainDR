import argparse
import bisect
import io
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

SUPPORTED = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA}
DOC_EXT = {".md", ".txt", ".csv", ".xlsx", ".json"}


class JoinedParts(io.RawIOBase):
    # reads the split parts as one file, so no joined copy is written to disk
    def __init__(self, paths):
        self.paths = paths
        self.sizes = [p.stat().st_size for p in paths]
        self.starts = [0]
        for s in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + s)
        self.total = sum(self.sizes)
        self.pos = 0
        self.handles = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.total + offset
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.total:
            return 0
        i = bisect.bisect_right(self.starts, self.pos) - 1
        if i not in self.handles:
            self.handles[i] = open(self.paths[i], "rb")
        f = self.handles[i]
        offset = self.pos - self.starts[i]
        f.seek(offset)
        data = f.read(min(len(buf), self.sizes[i] - offset))
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)

    def close(self):
        for f in self.handles.values():
            f.close()
        super().close()


def find_parts(folder):
    parts = sorted(folder.glob("MMRDR.zip.[0-9][0-9][0-9]"))
    if not parts:
        sys.exit(f"no MMRDR.zip.001 style files found in {folder}")
    numbers = [int(p.suffix[1:]) for p in parts]
    if numbers != list(range(1, len(parts) + 1)):
        sys.exit(f"parts are not a complete sequence: {[p.name for p in parts]}")
    empty = [p.name for p in parts if p.stat().st_size == 0]
    if empty:
        sys.exit(f"empty parts, download them again: {empty}")
    unfinished = list(folder.glob("MMRDR.zip.*.crdownload")) + list(folder.glob("MMRDR.zip.*.part"))
    if unfinished:
        sys.exit(f"downloads still in progress: {[p.name for p in unfinished]}")
    return parts


def show_contents(infos):
    groups = defaultdict(lambda: [0, 0])
    for info in infos:
        folder = "/".join(PurePosixPath(info.filename).parts[:-1][:3]) or "."
        groups[folder][0] += 1
        groups[folder][1] += info.file_size
    print("\n== Archive contents ==")
    for folder, (n, size) in sorted(groups.items())[:50]:
        print(f"{folder:<50} {n:>7} files {size / 1e9:7.2f} GB")
    if len(groups) > 50:
        print(f"... {len(groups) - 50} more folders not shown")

    docs = [i.filename for i in infos if PurePosixPath(i.filename).suffix.lower() in DOC_EXT]
    print("\n== Label and document files ==")
    for name in docs[:40]:
        print(" ", name)
    if len(docs) > 40:
        print(f"  ... {len(docs) - 40} more")


def cfp_folders(infos, out):
    found = set()
    for info in infos:
        parts = PurePosixPath(info.filename).parts
        for k, part in enumerate(parts[:-1]):
            if "cfp" in part.lower():
                found.add(out.joinpath(*parts[:k + 1]))
                break
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="folder holding MMRDR.zip.001 to .009")
    ap.add_argument("--out", default="data", help="where to extract")
    ap.add_argument("--all", action="store_true", help="extract UWF and OCT too")
    args = ap.parse_args()

    parts = find_parts(Path(args.parts))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"found {len(parts)} parts, {sum(p.stat().st_size for p in parts) / 1e9:.2f} GB in total")

    raw = JoinedParts(parts)
    try:
        zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
    except zipfile.BadZipFile as e:
        sys.exit(f"could not open the archive: {e}")

    infos = [i for i in zf.infolist() if not i.is_dir()]
    show_contents(infos)

    if args.all:
        chosen = infos
    else:
        chosen = [i for i in infos if "cfp" in i.filename.lower()
                  or PurePosixPath(i.filename).suffix.lower() in {".md", ".txt"}]
        if not any("cfp" in i.filename.lower() for i in chosen):
            sys.exit("\nno paths contain 'CFP', so nothing was extracted")

    bad = Counter(i.compress_type for i in chosen if i.compress_type not in SUPPORTED)
    if bad:
        sys.exit(f"\nunsupported compression methods {dict(bad)}, nothing extracted")
    if any(i.flag_bits & 0x1 for i in chosen):
        sys.exit("\narchive is password protected, nothing extracted")

    need = sum(i.file_size for i in chosen)
    free = shutil.disk_usage(out).free
    print(f"\nextracting {len(chosen)} files, {need / 1e9:.2f} GB ({free / 1e9:.1f} GB free)")
    if free < need * 1.05:
        sys.exit("not enough free disk space")

    for k, info in enumerate(chosen, 1):
        target = out / info.filename
        # skip files already extracted on an earlier run
        if not (target.exists() and target.stat().st_size == info.file_size):
            try:
                zf.extract(info, out)
            except (zipfile.BadZipFile, OSError) as e:
                # remove the half-written file so a rerun doesn't skip it
                target.unlink(missing_ok=True)
                sys.exit(f"failed on {info.filename}: {e}")
        if k % 500 == 0 or k == len(chosen):
            print(f"  {k}/{len(chosen)} files")

    zf.close()
    raw.close()
    for folder in cfp_folders(chosen, out):
        print(f"\nCFP data is in: {folder}")


if __name__ == "__main__":
    main()
