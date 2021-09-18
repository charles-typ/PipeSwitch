from queue import Queue
from multiprocessing import Process

import torch
import time

from pipeswitch.worker_common import ModelSummary
from pipeswitch.worker_terminate import WorkerTermThd
from util.util import timestamp

class WorkerProc(Process):
    def __init__(self, model_list, pipe, param_trans_pipe, term_pipe, wrid):
        super(WorkerProc, self).__init__()
        self.model_list = model_list
        self.pipe = pipe
        self.param_trans_pipe = param_trans_pipe
        self.term_pipe = term_pipe
        self.id = wrid

    def run(self):
        timestamp('worker', 'start')

        # Warm up CUDA and get shared cache
        torch.randn(1024, device='cuda')
        time.sleep(1)
        torch.cuda.recv_shared_cache() # pylint: disable=no-member
        timestamp('worker', 'share_gpu_memory')

        # Create requried variables
        model_map = {}
        TERMINATE_SIGNAL = [0] # 0 Idle, 1 Running, 2 Terminate
        complete_queue = Queue()

        # Import models
        for model_name in self.model_list:
            model_summary = ModelSummary(model_name,
                                         TERMINATE_SIGNAL,
                                         self.param_trans_pipe)
            model_map[hash(model_name)] = model_summary
        timestamp('worker', 'import models')

        # ------- start terminate thread -----------
        term_t = WorkerTermThd(self.term_pipe, complete_queue, TERMINATE_SIGNAL)
        term_t.start()
        timestamp('worker', 'start_term_thd')
        # ------- terminate thread started ---------

        count = 0
        while True:
            # event loop get a msg then compute
            # after started forward compute
            # last while loop for receiving complete queue trans
            agent, model_name = self.pipe.recv()
            if model_name is None:
                if model_summary is not None:
                    model_summary.reset_initialized(model_summary.model)
                continue

            model_summary = model_map[hash(model_name)]
            TERMINATE_SIGNAL[0] = 1
            timestamp('worker_proc', 'get_model')

            data_b = self.pipe.recv()
            #print("data recevied in worker is : ", data_b)
            timestamp('worker_proc', 'get_data')

            # start doing inference
            # frontend_scheduler will directly put
            # mod_list[0] in to self.complete_queue_trans
            try:
                #print("Worker id is: ", self.id)
                if self.id == 1:
                #    print("Send response 1 ", self.id)
                #if 'training' in model_name:
                    self.pipe.send('FNSH')
                    agent.send(b'FNSH')
                #print("Here 1", flush=True)

                with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                    #print("Here 2", flush=True)
                    output = model_summary.execute(data_b)
                    #print("Here 3", flush=True)
                    self.pipe.send('FNSH')
                    agent.send(b'FNSH')
                    print ('Get output', output)
                    del output
                #print("Here 4", flush=True)

                #if self.id == 0:
                #    print("Send response 2 ", self.id)
                #    self.pipe.send('FNSH')
                #    agent.send(b'FNSH')

                #if 'inference' in model_name:
                #    self.pipe.send('FNSH')
                #    agent.send(b'FNSH')
            except Exception as e:
                print("exception: at " , time.time(), flush=True)
                if str(e) == 'Invalid complete trans':
                    print("sending this response anyway: ", self.id)
                    self.pipe.send('FNSH')
                    agent.send(b'FNSH')
                else:
                    print(str(e))
                complete_queue.put('FNSH')


            #with torch.cuda.stream(model_summary.cuda_stream_for_computation):
            #    output = model_summary.execute(data_b)
            #    self.pipe.send('FNSH')
            #    agent.send(b'FNSH')
            #    print ('Get output', output)
            #    del output


            # start do cleaning
            TERMINATE_SIGNAL[0] = 0
            timestamp('worker_comp_thd', 'complete')

            #model_summary.reset_initialized(model_summary.model)
            #count = count + 1
            #print("Count is increasing: ", count)

