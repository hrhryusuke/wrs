import socket
import numpy as np
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 7003))
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

while True:
    msg = s.recv(960)
    # print(msg)
    # data = msg.decode()
    # print(data.split(' ')[:2], data.split(' ')[-1])
    # data = data.split(' ')[2:-1]
    # # for i in range(len(data)):
    # print(len(data), len(data)%3)
    # while True:
    #     data[]
    print(type(msg))
    print(len(msg))

    with open('/B4_Research/neuron_data.dat', 'wb') as f:
        f.write(msg)
        f.close()

    data_array = np.frombuffer(msg, dtype=np.float32)
    print(data_array)

    # d_msg = msg.decode()
    # d_msg_split = d_msg.split(' ')[:15*16]
    # print(len(d_msg_split))
    # print(d_msg_split)
    #
    # with open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data_string.txt', 'w') as f:
    #     f.write(d_msg)
    #     f.close()

    # print(len(d_msg_split))

    # for i in range(10):
    #     with open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data_perfect.txt', 'w') as f:
    #         f.write(f'{d_msg_split[i]}')
    #         f.close()
    #     if (i + 1) % 16 == 0:
    #         with open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data_perfect.txt', 'w') as f:
    #             f.write('\n')
    #             f.close()




    # with open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data_string.txt', 'w') as f3:
    #     f3.write(str(msg))
    #     f3.close()
    #
    # d_msg = msg.decode('ascii')
    #
    # with open('C:/Users/Yusuke Hirao/PycharmProjects/wrs/B4_Research/neuron_data_decode.txt', 'w') as f2:
    #     f2.write(d_msg)
    #     f2.close()

    break
