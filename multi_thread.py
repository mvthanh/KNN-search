import threading
import time
import numpy as np


class Name(object):
    named_id = dict()


def add_name_by_ids(id, name: Name):
    time.sleep(5)
    name.named_id[id] = 'haha {}'.format(id)


def call_thread(func, args):
    thread = threading.Thread(target=func, name='thread_{}'.format(time.time()),
                              args=args)
    thread.start()


if __name__ == '__main__':
    named_clt = Name()
    rand = np.random.randint(0, 10, 50)
    for i in rand:
        if i in named_clt.named_id.keys():
            name = named_clt.named_id[i]
        else:
            name = i
            call_thread(add_name_by_ids, [i, named_clt])
            # add_name_by_ids(i, named_id)
        print(name)
        time.sleep(1)



