import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp

def send_request(client, task_name, data):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize data
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    if data is not None:
        data_b = data.numpy().tobytes()
        length = len(data_b)
    else:
        data_b = None
        length = 0
    length_b = struct.pack('I', length)
    timestamp('client', 'after_inference_serialization')

    # Send Data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(length_b)
    client.send(data_b)
    timestamp('client', 'after_request_%s' % task_name)

def recv_response(client):
    reply_b = client.recv(4)
    reply = reply_b.decode()
    timestamp('client', 'after_reply')

def close_connection(client):
    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_connection')

def main():
    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])

    task_name_inf = '%s_inference' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    latency_list = []
    for k in range(20):

        # Connect
        num_parallel_request = 1
        client_list = []
        client_inf_2 = TcpClient('localhost', 12345)
        for i in range(num_parallel_request):
            client_inf = TcpClient('localhost', 12345)
            client_list.append(client_inf)
        for i in range(num_parallel_request):
            send_request(client_list[i], task_name_inf, data)

        if k == 0:
            time.sleep(2)
        # Send inference request
        time.sleep(500 / 1000)
        time_1 = time.time()
        send_request(client_inf_2, task_name_inf, data)

        # Recv inference reply
        recv_response(client_inf_2)
        time_2 = time.time()
        print(time_2)
        print(time_1)
        latency = (time_2 - time_1) * 1000
        print(latency)
        latency_list.append(latency)

        #time.sleep(1)
        for i in range(num_parallel_request):
            recv_response(client_list[i])
            close_connection(client_list[i])
        close_connection(client_inf_2)
        time.sleep(2)
        timestamp('**********', '**********')

    print()
    print()
    print()
    stable_latency_list = latency_list[10:]
    #stable_latency_list = latency_list
    print (stable_latency_list)
    print ('Latency: %f ms (stdev: %f)' % (statistics.mean(stable_latency_list),
                                           statistics.stdev(stable_latency_list)))

if __name__ == '__main__':
    main()
