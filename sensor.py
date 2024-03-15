import socket
import struct
import time
import numpy as np
import pandas as pd

# set connection info
UDP_IP = "0.0.0.0"
UDP_PORT = 5500
SO_TIMESTAMPNS = 35
SOF_TIMESTAMPING_RX_SOFTWARE = (1 << 3)

# generate filter for broken chunks
fx256 = np.ones(256, dtype=bool)
fx256[:6] = False
fx256[64*4-5:] = False
fx256[32] = False
fx256[96] = False
fx256[160] = False
fx256[224] = False

for i in range(1, 4):
    fx256[64*i-5:64*i+6] = False

# setup socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, SOF_TIMESTAMPING_RX_SOFTWARE)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# connect to socket
sock.bind((UDP_IP, UDP_PORT))
fd = sock.fileno()

print(f"Collecting from descriptor ID {fd}")

"""
    START RECORDING PACKETS
"""

start = time.time()
end = 60
prev = None
buffer = []
data_array = []

while (time.time() - start) < (end + 1):
    # receive data from socket
    try:
        data, ancdata, _, _ = sock.recvmsg(4096, 128)
    except socket.timeout:
        print("Socket Timeout")
        continue

    # check packet size
    if len(data) < 18:
        print("rx: packet too small")
        continue

    # collect data arrays
    data_array.append(np.frombuffer(data, dtype=np.int16))

    # process packet
    s, ns = struct.unpack('LL', ancdata[0][2])
    ts = s * 10**9 + ns

    ver, mask, rssi, fc, mac, _seq, conf, chanspec, chip = struct.unpack('<BBbB6sHHHH', data[:18])
    seq = _seq >> 4
    mac_str = mac.hex(':')
    count = (len(data) - 18) // 2
    data = np.frombuffer(data, dtype='<h', count=count, offset=18).astype(np.float32).view(np.complex64)

    # check data size
    assert len(data) == 256, f"Data of {len(data)} is not expected"

    # simple filter for broken chunks
    pl = data[fx256]
    x = pl.reshape((4,-1))
    mn = np.mean(np.abs(x), axis=-1)
    msk = np.zeros(mn.shape, dtype=bool)
    msk[np.argmax(mn)] = True
    pl  = x[msk]
    pl = pl.ravel()

    v = np.abs(pl)
    maxv = np.max(v)
    if maxv != 0:
        v /= maxv

    # calculate motion
    if prev is not None:
        motion = (np.corrcoef(v, prev)[0][1])**2
        motion = -10 * motion + 10
        buffer.append(dict(time=ts, motion=motion, rssi=rssi,mac=mac_str, seq=seq))
    prev = v

    # progress bar
    bar_length = 50
    progress = int(time.time() - start)
    progress = min(end, max(0, progress))
    num_blocks = int(round(bar_length * progress / end))
    bar = '█' * num_blocks + '-' * (bar_length - num_blocks)
    print(f'\r[{bar}] {(num_blocks * 2)}%', end='', flush=True)

print("\n")
sock.close()

# save collected data
filename = f"collected_csi_{int(time.time())}"

# save time series data
df = pd.DataFrame(buffer)
df.to_csv(f"./time_series/{filename}.csv", index=False)

# ensure that data array sizes are equal
padded_array = []
for array in data_array:
    if len(array) > 521:
        padded_array.append(array[:521])
    elif len(array) < 521:
        padded_array.append(np.pad(array, (0, 521 - len(array)), 'constant', constant_values=0))
    else:
        padded_array.append(array)

# convert to numpy array
data_array = np.array(padded_array)

# save data arrays
np.save(f"./buffers/{filename}.npy", data_array)
