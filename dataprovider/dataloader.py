import torch, random

random.seed()


class WeightedSamplingDataloader:

    def __init__(self, dataset_list, weight_list, batch_size):

        assert len(dataset_list) == len(weight_list)

        self.dataset_list = dataset_list
        self.weight_list = weight_list
        self.batch_size = batch_size

        self.dataset_iter_list = []

        total_data_size = 0

        for d in dataset_list:
            total_data_size += len(d)

            self.dataset_iter_list.append(iter(d))
        
        self.total_data_size = total_data_size
        self.dataset_index_list = list(range(len(dataset_list)))

    def __iter__(self):
        self.num = 0
        return self

    def __len__(self):
        return self.total_data_size // self.batch_size

    def __next__(self):
        if self.num >= len(self):
            raise StopIteration()
        else:
            self.num +=1
        
        # fetch batch
        # get random dataset index sample

        dataset_index_list = random.choices(self.dataset_index_list, weights=self.weight_list, k=self.batch_size)

        fetch_list = []

        for i in dataset_index_list:
            a = self.dataset_iter_list[i].next()
            fetch_list.append(a)
        
        if isinstance(fetch_list[0], tuple):
            
            fetch_item_len = len(fetch_list[0])
        else:
            fetch_item_len = 1

        result = []
        for _ in range(fetch_item_len):

            result.append(list())

        for a in fetch_list:
            if fetch_item_len ==1:
                result[0].append(a)
            else:
                for j, b in enumerate(a):
                    result[j].append(b)
        

        if fetch_item_len==1:
            result = result[0]
        
        return result

