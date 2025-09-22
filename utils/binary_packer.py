import struct
import pickle
from io import BytesIO
from typing import Any, Union

class BinaryPacker:
  """
  Packs and unpacks nested structures into a binary format.
  Supports dicts, lists, strings, bytes, numbers.
  """

  def __init__(self):
    self.TYPE_MAP = {
      dict:  b'D',
      list:  b'L',
      str:   b'S',
      bytes: b'B',
      int:   b'I',
      float: b'F',
    }

  def pack(self, obj: Any) -> bytes:
    """
    Recursively pack a Python object into binary format.
    """
    buffer = BytesIO()
    self._pack_obj(buffer, obj)
    return buffer.getvalue()

  def unpack(self, data: bytes) -> Any:
    """
    Recursively unpack binary data into Python object.
    """
    buffer = BytesIO(data)
    return self._unpack_obj(buffer)

  def _pack_obj(self, buf: BytesIO, obj: Any):
    t = type(obj)

    if t == dict:
      buf.write(self.TYPE_MAP[dict])
      buf.write(struct.pack('I', len(obj)))
      for key, val in obj.items():
        self._pack_obj(buf, key)
        self._pack_obj(buf, val)

    elif t == list:
      buf.write(self.TYPE_MAP[list])
      buf.write(struct.pack('I', len(obj)))
      for item in obj:
        self._pack_obj(buf, item)

    elif t == str:
      data = obj.encode('utf-8')
      buf.write(self.TYPE_MAP[str])
      buf.write(struct.pack('I', len(data)))
      buf.write(data)

    elif t == bytes:
      buf.write(self.TYPE_MAP[bytes])
      buf.write(struct.pack('I', len(obj)))
      buf.write(obj)

    elif t == int:
      buf.write(self.TYPE_MAP[int])
      buf.write(struct.pack('q', obj))

    elif t == float:
      buf.write(self.TYPE_MAP[float])
      buf.write(struct.pack('d', obj))

    else:
      # fallback to pickle for unknown types
      print('[WARNING] Unknow type, using pickle')
      buf.write(b'P')  # Pickle marker
      pdata = pickle.dumps(obj)
      buf.write(struct.pack('I', len(pdata)))
      buf.write(pdata)

  def _unpack_obj(self, buf: BytesIO) -> Any:
    t = buf.read(1)

    if t == b'D':
      length = struct.unpack('I', buf.read(4))[0]
      return {self._unpack_obj(buf): self._unpack_obj(buf) for _ in range(length)}

    elif t == b'L':
      length = struct.unpack('I', buf.read(4))[0]
      return [self._unpack_obj(buf) for _ in range(length)]

    elif t == b'S':
      length = struct.unpack('I', buf.read(4))[0]
      return buf.read(length).decode('utf-8')

    elif t == b'B':
      length = struct.unpack('I', buf.read(4))[0]
      return buf.read(length)

    elif t == b'I':
      return struct.unpack('q', buf.read(8))[0]

    elif t == b'F':
      return struct.unpack('d', buf.read(8))[0]

    elif t == b'P':
      length = struct.unpack('I', buf.read(4))[0]
      return pickle.loads(buf.read(length))

    else:
      raise ValueError(f"Unknown type marker: {t}")

if __name__ == "__main__":
  packer = BinaryPacker()

  nested = {
    "image1": b'\x89PNG...',  # image bytes
    "meta": {
      "label": "cat",
      "confidence": 0.98
    },
    "flags": [True, False, "maybe"]
  }

  packed = packer.pack(nested)
  restored = packer.unpack(packed)

  import pdb;pdb.set_trace()

  '''
  (Pdb) len(packed)
  127
  (Pdb) pickle.dumps(nested)
  b'\x80\x04\x95]\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x06image1\x94C\x07\x89PNG...\x94\x8c\x04meta\x94}\x94(\x8c\x05label\x94\x8c\x03cat\x94\x8c\nconfidence\x94G?\xef\\(\xf5\xc2\x8f\\u\x8c\x05flags\x94]\x94(\x88\x89\x8c\x05maybe\x94eu.'
  (Pdb) len(pickle.dumps(nested))
  104
  '''
