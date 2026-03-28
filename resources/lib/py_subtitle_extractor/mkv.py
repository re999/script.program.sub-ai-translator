from .ebml import read_id, read_size, read_vint
from typing import List, Tuple, BinaryIO, Optional, Callable, Union
import mmap

SEGMENT    = 0x18538067
TRACKS     = 0x1654AE6B
TRACKENTRY = 0xAE
TRACKTYPE  = 0x83
TRACKNUMBER= 0xD7
CODECID    = 0x86
TRACKNAME  = 0x536E
LANG_BCP47 = 0x22B59D
LANG       = 0x22B59C
CLUSTER    = 0x1F43B675
TIMECODE   = 0xE7
SIMPLEBLK  = 0xA3
BLKGROUP   = 0xA0
BLOCK      = 0xA1
BLOCKDUR   = 0x9B

def _read_header(f: BinaryIO) -> Tuple[int,int]:
    eid,_ = read_id(f)
    sz,_  = read_size(f)
    return eid, sz

def extract_subtitle_tracks(path: str) -> List[dict]:
    from io import BytesIO
    with open(path, "rb") as f:
        _, h = _read_header(f); f.seek(h,1)
        eid, h = _read_header(f)
        if eid != SEGMENT:
            return []
        end = f.tell() + h
        while f.tell()<end:
            eid, sz = _read_header(f)
            if eid == TRACKS:
                return [t for t in _parse_tracks(f.read(sz)) if t["type"]==0x11]
            f.seek(sz,1)
    return []

def _parse_tracks(data: bytes) -> List[dict]:
    from io import BytesIO
    buf = BytesIO(data); out=[]
    while True:
        try: eid, sz = _read_header(buf)
        except EOFError: break
        chunk = buf.read(sz)
        if eid==TRACKENTRY:
            out.append(_parse_track_entry(chunk))
    return out

def _parse_track_entry(data: bytes) -> dict:
    from io import BytesIO
    buf = BytesIO(data)
    info={"type":0,"track_number":0,"codec_id":"","language":"","name":""}
    while True:
        try: eid, sz = _read_header(buf)
        except EOFError: break
        v=buf.read(sz)
        if   eid==TRACKTYPE:   info["type"]=v[0]
        elif eid==TRACKNUMBER: info["track_number"]=int.from_bytes(v,"big")
        elif eid==CODECID:     info["codec_id"]=v.decode("utf-8","ignore")
        elif eid==LANG_BCP47:  info["language"]=v.decode("utf-8","ignore")
        elif eid==LANG and not info["language"]:
                                info["language"]=v.decode("utf-8","ignore")
        elif eid==TRACKNAME:   info["name"]=v.decode("utf-8","ignore")
    return info

import mmap

def extract_subtitles(
    path: str,
    track: int,
    on_progress: Optional[Callable[[float], None]] = None
) -> List[Union[Tuple[int, str], Tuple[int, str, int]]]:
    subs = []
    try:
        with open(path, "rb") as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as f:
                return _extract_subtitles_from(f, track, on_progress)
    except (ValueError, OSError):
        with open(path, "rb") as f:
            return _extract_subtitles_from(f, track, on_progress)
    return subs

def _extract_subtitles_from(
    f: BinaryIO,
    track: int,
    on_progress: Optional[Callable[[float], None]] = None
) -> List[Union[Tuple[int, str], Tuple[int, str, int]]]:
    subs = []
    _, h = _read_header(f); f.seek(h, 1)
    eid, h = _read_header(f)
    if eid != SEGMENT:
        return subs

    segment_start = f.tell()
    segment_end = segment_start + h
    last_percent = -1

    while f.tell() < segment_end:
        eid, sz = _read_header(f)
        if eid == CLUSTER:
            subs += _parse_cluster(f, sz, track)
        else:
            f.seek(sz, 1)

        if on_progress:
            pos = f.tell() - segment_start
            percent = int(100 * pos / (segment_end - segment_start))
            if percent != last_percent and percent % 1 == 0:
                last_percent = percent
                on_progress(percent / 100)

    return subs

def _parse_cluster(f: BinaryIO, size: int, track: int) -> List[Union[Tuple[int, str], Tuple[int, str, int]]]:
    end=f.tell()+size
    base=_read_cluster_time(f,end)
    out=[]
    while f.tell()<end:
        eid, sz = _read_header(f)
        if eid==SIMPLEBLK:
            out+=_handle_block(f,sz,base,track)
        elif eid==BLKGROUP:
            out+=_handle_group(f,sz,base,track)
        else:
            f.seek(sz,1)
    return out

def _read_cluster_time(f: BinaryIO, end: int) -> int:
    while f.tell()<end:
        eid, sz = _read_header(f)
        if eid==TIMECODE:
            return int.from_bytes(f.read(sz),"big")
        f.seek(sz,1)
    return 0

def _handle_block(f: BinaryIO, sz: int, base: int, track: int) -> List[Tuple[int,str]]:
    trk,vlen=read_vint(f)
    t=int.from_bytes(f.read(2),"big",signed=True)
    f.read(1)
    plen=sz-vlen-3
    if trk==track:
        txt=f.read(plen).decode("utf-8","ignore").strip()
        return [(base+t,txt)]
    f.seek(plen,1)
    return []

def _handle_group(f: BinaryIO, sz: int, base: int, track: int) -> List[Union[Tuple[int, str], Tuple[int, str, int]]]:
    end=f.tell()+sz
    txt = None
    t_rel = None
    duration = None
    while f.tell()<end:
        eid,s = _read_header(f)
        if eid==BLOCK:
            pos = f.tell()
            trk,vlen=read_vint(f)
            t_rel=int.from_bytes(f.read(2),"big",signed=True)
            f.read(1)
            plen=s-vlen-3
            if trk==track:
                txt=f.read(plen).decode("utf-8","ignore").strip()
            else:
                f.seek(pos+s)
        elif eid==BLOCKDUR:
            duration = int.from_bytes(f.read(s), "big")
        else:
            f.seek(s,1)
    if txt and t_rel is not None:
        start = base + t_rel
        if duration:
            return [(start, txt, start + duration)]
        return [(start, txt)]
    return []
