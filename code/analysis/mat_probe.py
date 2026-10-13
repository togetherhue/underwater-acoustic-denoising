"""Minimal MATLAB v5 .mat reader (pure python, stdlib only) for quick inspection."""
import struct, sys, zlib, os

DT = {1: 'int8', 2: 'uint8', 3: 'int16', 4: 'uint16', 5: 'int32', 6: 'uint32',
      7: 'single', 9: 'double', 12: 'int64', 13: 'uint64', 14: 'matrix',
      15: 'compressed', 16: 'utf8', 17: 'utf16', 18: 'utf32'}
CLS = {1: 'cell', 2: 'struct', 3: 'object', 4: 'char', 5: 'sparse', 6: 'double',
       7: 'single', 8: 'int8', 9: 'uint8', 10: 'int16', 11: 'uint16', 12: 'int32',
       13: 'uint32', 14: 'int64', 15: 'uint64', 16: 'function'}
FMT = {'double': 'd', 'single': 'f', 'int16': 'h', 'uint16': 'H',
       'int32': 'i', 'uint32': 'I', 'int8': 'b', 'uint8': 'B',
       'int64': 'q', 'uint64': 'Q'}
SZ = {'double': 8, 'single': 4, 'int16': 2, 'uint16': 2, 'int32': 4,
      'uint32': 4, 'int8': 1, 'uint8': 1, 'int64': 8, 'uint64': 8}


def pad8(n):
    return (8 - n % 8) % 8


def read_tag(b, off):
    """Return (dtype, nbytes, data_offset, next_element_offset)."""
    t16, s16 = struct.unpack_from('<HH', b, off)
    if s16 != 0:                       # small data element format
        return t16, s16, off + 4, off + 8
    dt, nb = struct.unpack_from('<II', b, off)
    return dt, nb, off + 8, off + 8 + nb + pad8(nb)


def decode_numeric(b, dt, nb, doff):
    name = DT.get(dt, dt)
    if name not in FMT or nb == 0:
        return name, ()
    cnt = nb // SZ[name]
    if name == 'double':
        vals = struct.unpack_from('<%dd' % cnt, b, doff)
    elif name == 'single':
        vals = struct.unpack_from('<%df' % cnt, b, doff)
    else:
        vals = struct.unpack_from('<%d%s' % (cnt, FMT[name]), b, doff)
    return name, vals


def parse_matrix(b, off):
    """Parse one miMATRIX body starting at off. Returns (info, next_off)."""
    info = {}
    dt, nb, doff, noff = read_tag(b, off)          # Array flags
    if dt in (6, 5) and nb >= 8:
        flags = struct.unpack_from('<I', b, doff)[0]
    elif dt in (4, 3):
        flags = struct.unpack_from('<H', b, doff)[0]
    else:
        flags = 0
    info['class'] = CLS.get(flags & 0xFF, flags & 0xFF)
    off = noff

    dt, nb, doff, noff = read_tag(b, off)          # Dimensions
    _, dv = decode_numeric(b, dt, nb, doff)
    info['dims'] = tuple(int(v) for v in dv) if dv else ()
    off = noff

    dt, nb, doff, noff = read_tag(b, off)          # Name
    info['name'] = bytes(b[doff:doff + nb]).decode('utf-8', 'replace') if nb else ''
    off = noff

    if info['class'] in ('cell', 'struct', 'object'):
        info['data'] = ()
        info['children'] = []
        n = 1
        for d in info['dims']:
            n *= d
        if info['class'] == 'struct':
            # field name length + field names, then values
            dt, nb, doff, noff = read_tag(b, off)
            flen = struct.unpack_from('<h', b, doff)[0] if nb else 0
            off = noff
            dt, nb, doff, noff = read_tag(b, off)
            names = []
            for i in range(nb // flen if flen else 0):
                nm = bytes(b[doff + i * flen:doff + (i + 1) * flen]).split(b'\0')[0]
                names.append(nm.decode('utf-8', 'replace'))
            info['fields'] = names
            off = noff
            for _ in range(n * max(len(names), 1)):
                if off >= len(b) - 8:
                    break
                child, off = parse_matrix(b, off)
                info['children'].append(child)
        else:
            for _ in range(n):
                if off >= len(b) - 8:
                    break
                child, off = parse_matrix(b, off)
                info['children'].append(child)
        return info, off

    dt, nb, doff, noff = read_tag(b, off)          # Numeric payload
    name, vals = decode_numeric(b, dt, nb, doff)
    info['dtype'] = name
    info['data'] = vals
    return info, noff


def load(path):
    with open(path, 'rb') as f:
        b = f.read()
    off = 128
    out = []
    while off < len(b) - 8:
        dt, nb, doff, noff = read_tag(b, off)
        if dt == 15:
            inner = zlib.decompress(bytes(b[doff:doff + nb]))
            p = 0
            while p < len(inner) - 8:
                idt, inb, idoff, inoff = read_tag(inner, p)
                if idt != 14:
                    break
                m, _ = parse_matrix(inner, idoff)
                out.append(m)
                p = inoff
        elif dt == 14:
            m, _ = parse_matrix(b, doff)
            out.append(m)
        off = noff
    return out


def stat(vals):
    n = len(vals)
    if n == 0:
        return ''
    mn, mx = min(vals), max(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    return 'min=%.6g max=%.6g mean=%.6g std=%.6g' % (mn, mx, mean, std)


def summarize(path, show_head=True):
    print('=' * 72)
    print('FILE: %s   (%.1f KB)' % (path, os.path.getsize(path) / 1024))
    for m in load(path):
        d = m.get('data', ())
        print('  "%s"  class=%s dtype=%s dims=%s n=%d' %
              (m.get('name'), m.get('class'), m.get('dtype'), m.get('dims'), len(d)))
        if d:
            print('      ' + stat(d))
            if show_head:
                print('      head=%s' % ['%.5g' % v for v in d[:6]])
        if 'children' in m and m['children']:
            for c in m['children']:
                cd = c.get('data', ())
                print('      - child "%s" class=%s dims=%s n=%d %s' %
                      (c.get('name'), c.get('class'), c.get('dims'), len(cd), stat(cd)))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        summarize(p)
