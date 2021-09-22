import lzma

import tensorflow as tf

class BenchmarkTrace:
    '''
    Benchmark trace parsing class
    '''

    def __init__(self):
        self.pc_mapping = {'oov': 0}
        self.page_mapping = {'oov': 0}
        self.data = []
        self.inst_ids = []

    def read_file(self, f):
        '''
        Reads and processes the data in the benchmark trace files
        '''
        for i, line in enumerate(f):
            # Necessary for some extraneous lines in MLPrefetchingCompetition traces
            if line.startswith('***') or line.startswith('Read'):
                continue
            inst_id, pc, addr = self.process_line(line)
            self.process_row(i, inst_id, pc, addr)
        self.data = tf.convert_to_tensor(self.data)

    def process_row(self, idx, inst_id, pc, addr):
        '''
        Process PC / Address

        TODO: Handle the pc localization
        '''
        page, offset = addr >> 12, (addr >> 6) & 0x3f

        if pc not in self.pc_mapping:
            self.pc_mapping[pc] = len(self.pc_mapping)

        if page not in self.page_mapping:
            self.page_mapping[page] = len(self.page_mapping)

        self.data.append([self.pc_mapping[pc], self.page_mapping[page], offset])
        self.inst_ids.append(inst_id)

    def process_line(self, line):
        # File format for ML Prefetching Competition
        # Uniq Instr ID, Cycle Count,   Load Address,      PC of Load,        LLC Hit or Miss
        # int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

        # Return PC and Load Address
        split = line.strip().split(', ')
        return (int(split[0]), int(split[3], 16), int(split[2], 16))

    def num_pcs(self):
        return len(self.pc_mapping)

    def num_pages(self):
        return len(self.page_mapping)

    def _split_idx(self, config):
        # Computing end of train and validation splits
        train_split = int(config.train_split * (len(self.data) - config.sequence_length)) + config.sequence_length
        valid_split = int((config.train_split + config.valid_split) * (len(self.data) - config.sequence_length)) + config.sequence_length

        return train_split, valid_split

    def split(self, config, start_epoch, start_step):
        '''
        Splits the trace data into train / valid / test datasets
        '''
        train_split, valid_split = self._split_idx(config)

        def mapper(idx):
            '''
            Maps index in dataset to x = [pc_hist, page_hist, offset_hist], y = [page_target, offset_target]
            If sequence, page_target / offset_target is also a sequence

            Given a batch (x, y) where the first dimension corresponds to the batch
            pc_hist = x[:, :seq_len], page_hist = x[:, seq_len:2 * seq_len], offset_hist = x[:, 2 * seq_len:]
            page_target = y[:, 0], offset_target = y[:, 1]

            If sequence:
            page_target = y[:, 0, :], offset_target = y[:, 1, :]
            where the third axis / dimension is the time dimension
            '''
            start, end = idx - config.sequence_length, idx

            page_hist = self.data[start:end, 1]
            offset_hist = self.data[start:end, 2]

            # Slide the pc_hist to the left by 1 if using current PC
            if config.use_current_pc:
                pc_hist = self.data[start + 1:end + 1, 0]
            else:
                pc_hist = self.data[start:end, 0]

            # Return full target sequence if using sequence loss
            if config.sequence_loss:
                return tf.concat([pc_hist, page_hist, offset_hist], axis=-1), tf.cast([self.data[start + 1:end + 1, 1], self.data[start + 1:end + 1, 2]], "int64")
            else:
                return tf.concat([pc_hist, page_hist, offset_hist], axis=-1), tf.cast([self.data[end, 1], self.data[end, 2]], "int64")

        # Put together the datasets
        epoch_size = config.steps_per_epoch * config.batch_size
        def random(x):
            epoch = x // epoch_size
            step = x % epoch_size
            return tf.random.stateless_uniform((), minval=config.sequence_length, maxval=train_split, seed=(epoch, step), dtype=tf.dtypes.int64)

        # Needs to go to num_epochs + 1 because epochs is 1 indexed
        train_ds = (tf.data.Dataset
            .range(start_epoch * epoch_size + start_step * config.batch_size, (config.num_epochs + 1) * config.steps_per_epoch * config.batch_size)
            .map(random)
            .map(mapper)
            .batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        )

        valid_ds = (tf.data.Dataset.range(train_split, valid_split)
            .map(mapper)
            .batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        )

        test_ds = (tf.data.Dataset.range(valid_split, len(self.data))
            .map(mapper)
            .batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        )

        self.test_inst_ids = self.inst_ids[valid_split:]
        self.reverse_page_mapping = {v: k for k, v in self.page_mapping.items()}

        return train_ds, valid_ds, test_ds

def read_benchmark_trace(benchmark_path):
    '''
    Reads and processes the trace for a benchmark
    '''
    benchmark = BenchmarkTrace()

    if benchmark_path.endswith('.txt'):
        with open(benchmark_path, 'r') as f:
            benchmark.read_file(f)
    elif benchmark_path.endswith('.txt.xz'):
        with lzma.open(benchmark_path, mode='rt', encoding='utf-8') as f:
            benchmark.read_file(f)

    return benchmark
