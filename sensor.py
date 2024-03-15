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

print(f"Descriptor ID {fd}")

"""
    START RECORDING PACKETS
"""

start = time.time()
prev = None
buffer = []

while (time.time() - start) < 60:
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

    # logger
    eta = (time.time() - start) - 60
    print(f"ETA: {int(eta)}")

sock.close()

# save time series data
df = pd.DataFrame(buffer)
df.to_csv(f"./time_series/collected_csi_{int(time.time())}.csv", index=False)
