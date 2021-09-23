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
    #time_1 = time.time()
    client.send(task_name_length_b)
    #time_2 = time.time()
    client.send(task_name_b)
    #time_3 = time.time()
    client.send(length_b)
    #time_4 = time.time()
    client.send(data_b)
    #time_5 = time.time()
    timestamp('client', 'after_request_%s' % task_name)
    #print("Check time: 1 ", time_2 - time_1)
    #print("Check time: 2 ", time_3 - time_2)
    #print("Check time: 3 ", time_4 - time_3)
    #print("Check time: 4 ", time_5 - time_4)

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
        time.sleep(30 / 1000) # 56 16 10 8 16
        #time.sleep(50 / 1000) # 360 16 32  56 32
        #time.sleep(200 / 1000) # 360 64
        # time.sleep(350 / 1000) # 360 128
        #time.sleep(5 / 1000) # less than 16
        time_1 = time.time()
        send_request(client_inf_2, task_name_inf, data)

        # Recv inference reply
        recv_response(client_inf_2)
        time_2 = time.time()
        print(time_2, flush=True)
        print(time_1, flush=True)
        latency = (time_2 - time_1) * 1000
        print(latency, flush=True)
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
