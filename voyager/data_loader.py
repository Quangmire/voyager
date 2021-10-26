import lzma

import tensorflow as tf

from voyager.utils import timefunction

class BenchmarkTrace:
    '''
    Benchmark trace parsing class
    '''

    def __init__(self, config):
        self.config = config
        self.pc_mapping = {'oov': 0}
        self.page_mapping = {'oov': 0}
        self.reverse_page_mapping = {}
        self.data = [[0, 0, 0, 0, 0]]
        self.pc_data = [[]]
        self.online_cutoffs = []
        # Boolean value indicating whether or not to use the multiple labeling scheme
        self.offset_mask = (1 << config.offset_bits) - 1
        # Stores pc localized streams
        self.pc_addrs = {}
        self.pc_addrs_idx = {}
        # Counts number of occurences for low frequency addresses for delta prediction
        self.count = {}
        # Stores address localized streams for spatial localization
        self.cache_lines = {}
        self.cache_lines_idx = {}

    def read_and_process_file(self, f):
        self._read_file(f)
        self.reverse_page_mapping = {v: k for k, v in self.page_mapping.items()}
        if self.config.multi_label:
            self._generate_multi_label()
            # Needs to be regenerated to include deltas
            self.reverse_page_mapping = {v: k for k, v in self.page_mapping.items()}
        self._tensor()

    @timefunction('Tensoring data')
    def _tensor(self):
        self.data = tf.convert_to_tensor(self.data)
        if self.config.multi_label:
            self.pages = tf.convert_to_tensor(self.pages)
            self.offsets = tf.convert_to_tensor(self.offsets)
        self.pc_data = tf.ragged.constant(self.pc_data)

    @timefunction('Reading in data')
    def _read_file(self, f):
        '''
        Reads and processes the data in the benchmark trace files
        '''
        cur_phase = 1
        phase_size = 50 * 1000 * 1000
        for i, line in enumerate(f):
            # Necessary for some extraneous lines in MLPrefetchingCompetition traces
            if line.startswith('***') or line.startswith('Read'):
                continue
            inst_id, pc, addr = self.process_line(line)
            # We want 1 epoch per 50M instructions
            # TODO: Do we want to do every 50M instructions or 50M load instructions?
            if inst_id >= cur_phase * phase_size:
                self.online_cutoffs.append(i)
                cur_phase += 1
            self.process_row(i, inst_id, pc, addr)

    @timefunction('Generating multi-label data')
    def _generate_multi_label(self):
        # Previous address for delta localization
        prev_addr = {}
        if self.config.global_output:
            prev_addr['global'] = None
        self.pages = []
        self.offsets = []
        # 1 Global + 1 Delta + 1 PC + 1 Spatial + up to 10 Co-occurrence
        width = 14
        for i, (inst_id, mapped_pc, mapped_page, offset) in enumerate(self.data):
            mapped_pages = []
            offsets = []

            # Cache line of next load
            cur_addr = (self.reverse_page_mapping[mapped_page] << self.config.offset_bits) + offset

            # DELTAS FOR INFREQUENT ADDRESSES
            if self.count[(mapped_page, offset)] <= 2:
                # First spot for global which is unused here
                mapped_pages.append(-1)
                offsets.append(-1)
                # Only do delta if we're not the first memory address
                if (self.config.pc_localized and mapped_pc in prev_addr) or (self.config.global_stream and prev_addr['global'] is not None):
                    if self.config.pc_localized:
                        dist = cur_addr - prev_addr[mapped_pc]
                    else:
                        dist = cur_addr - prev_addr['global']
                    # Only do deltas for pages within 256 pages, which as of right now
                    # experimentally looks like it gives good coverage without unnecessarily
                    # blowing up the vocabulary size
                    if 0 <= (abs(dist) >> self.config.offset_bits) <= 256:
                        dist_page = None
                        dist_offset = abs(dist) & self.offset_mask
                        if dist > 0:
                            dist_page = '+' + str(abs(dist) >> self.config.offset_bits)
                        elif dist < 0:
                            dist_page = '-' + str(abs(dist) >> self.config.offset_bits)

                        # We don't care about when the next address was the previous one
                        if dist_page is not None:
                            if dist_page not in self.page_mapping:
                                self.page_mapping[dist_page] = len(self.page_mapping)

                            mapped_pages.append(self.page_mapping[dist_page])
                            offsets.append(dist_offset)
                    # Want to associate second spot with delta
                    else:
                        mapped_pages.append(-1)
                        offsets.append(-1)
                # Want to associate second spot with delta
                else:
                    mapped_pages.append(-1)
                    offsets.append(-1)
            # ORIGINAL OTHERWISE
            else:
                # First spot with global
                mapped_pages.append(mapped_page)
                offsets.append(offset)
                # Second spot for delta which is unused here
                mapped_pages.append(-1)
                offsets.append(-1)

            if not self.config.global_output:
                mapped_pages.append(-1)
                offsets.append(-1)
            else:
                # PC LOCALIZATION
                if self.pc_addrs_idx[mapped_pc] < len(self.pc_addrs[mapped_pc]) - 1:
                    self.pc_addrs_idx[mapped_pc] += 1
                    pc_page, pc_offset = self.pc_addrs[mapped_pc][self.pc_addrs_idx[mapped_pc]]
                    mapped_pages.append(pc_page)
                    offsets.append(pc_offset)
                # Want to associate third spot with pc localization
                else:
                    mapped_pages.append(-1)
                    offsets.append(-1)

            # SPATIAL LOCALIZATION
            # TODO: Possibly need to examine this default distance value to make sure that it
            #       isn't too small (or large) for accessing the max temporal distance
            for (_, _, spatial_page, spatial_offset) in self.data[i:i + 10]:
                if 0 < (1 << self.config.offset_bits) * (spatial_page - mapped_page) + (spatial_offset - offset) < 257:
                    mapped_pages.append(spatial_page)
                    offsets.append(spatial_offset)
                    break
            # Want to associate the fourth spot with spatial localization
            else:
                mapped_pages.append(-1)
                offsets.append(-1)

            # CO-OCCURRENCE
            # Compute frequency of pages in the future
            # TODO: Make sure that we want to do page instead of cache line.
            #       I tried using the cache line, but across the handful of
            #       benchmarks that I tested it on, there were maybe 30ish cases
            #       where the exact same cache line showed up more than once.
            #       - One possibility is that we might want to find the most
            #         occurring cache line given a history (of 1 or more) of the
            #         last loads throughout the trace.
            freq = {}
            best = 0
            for (_, _, co_page, co_offset) in self.data[i:i + 10]:
                tag = co_page
                if tag not in freq:
                    freq[tag] = 0
                freq[tag] += 1
                best = max(best, freq[tag])

            # Only want most frequent if it appears more than once, otherwise
            # co-occurrence would be slightly meaningless
            if best >= 2:
                # Take the most frequent cache lines
                for (_, _, co_page, co_offset) in self.data[i:i + 10]:
                    if freq[co_page] == best:
                        mapped_pages.append(co_page)
                        offsets.append(co_offset)

            # Save for delta
            if self.config.pc_localized:
                prev_addr[mapped_pc] = cur_addr
            else:
                prev_addr['global'] = cur_addr

            # Working with rectangular tensors is infinitely more painless than
            # Tensorflow's ragged tensors that have a bunch of unsupported ops
            for i in range(len(mapped_pages), width):
                mapped_pages.append(-1)
                offsets.append(-1)

            # Append the final list of pages and offsets. There will be some
            # duplicates, but that's okay.
            self.pages.append(mapped_pages)
            self.offsets.append(offsets)

        # No longer need these, might as well free it up from memory
        del self.pc_addrs
        del self.pc_addrs_idx
        del self.cache_lines
        del self.cache_lines_idx
        del self.count

    def process_row(self, idx, inst_id, pc, addr):
        '''
        Process PC / Address

        TODO: Handle the pc localization
        '''
        cache_line = addr >> 6
        page, offset = cache_line >> self.config.offset_bits, cache_line & self.offset_mask

        if pc not in self.pc_mapping:
            self.pc_mapping[pc] = len(self.pc_mapping)
            # These are needed for PC localization
            self.pc_addrs[self.pc_mapping[pc]] = []
            self.pc_addrs_idx[self.pc_mapping[pc]] = 0
            self.pc_data.append([0 for _ in range(self.config.sequence_length + self.config.prediction_depth)])

        if page not in self.page_mapping:
            self.page_mapping[page] = len(self.page_mapping)

        # Needed for delta localization
        if (self.page_mapping[page], offset) not in self.count:
            self.count[(self.page_mapping[page], offset)] = 0
        self.count[(self.page_mapping[page], offset)] += 1

        self.pc_addrs[self.pc_mapping[pc]].append((self.page_mapping[page], offset))

        # Needed for spatial localization
        if cache_line not in self.cache_lines:
            self.cache_lines[cache_line] = []
            self.cache_lines_idx[cache_line] = 0
        self.cache_lines[cache_line].append(idx)

        # Include the instruction ID for generating the prefetch file for running
        # in the ML-DPC modified version of ChampSim.
        # See github.com/Quangmire/ChampSim
        self.data.append([inst_id, self.pc_mapping[pc], self.page_mapping[page], offset, len(self.pc_data[self.pc_mapping[pc]])])
        self.pc_data[self.pc_mapping[pc]].append(len(self.data) - 1)

    def process_line(self, line):
        # File format for ML Prefetching Competition
        # See github.com/Quangmire/ChampSim
        # Uniq Instr ID, Cycle Count,   Load Address,      PC of Load,        LLC Hit or Miss
        # int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

        # Return Inst ID, PC, and Load Address
        split = line.strip().split(', ')
        return (int(split[0]), int(split[3], 16), int(split[2], 16))

    def num_pcs(self):
        return len(self.pc_mapping)

    def num_pages(self):
        return len(self.page_mapping)

    def _split_idx(self):
        # Computing end of train and validation splits
        train_split = int(self.config.train_split * (len(self.data) - self.config.sequence_length)) + self.config.sequence_length
        valid_split = int((self.config.train_split + self.config.valid_split) * (len(self.data) - self.config.sequence_length)) + self.config.sequence_length
        print('TEST INST ID STARTS AROUND ~', self.data[valid_split, 0])

        return train_split, valid_split

    def split(self, start_epoch=0, start_step=0, online=False, start_phase=0):
        '''
        Splits the trace data into train / valid / test datasets
        '''
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
            hists = []

            # Global Stream Input and Output
            start, end = idx - self.config.sequence_length - self.config.prediction_depth, idx - self.config.prediction_depth
            pred_start, pred_end = idx - self.config.sequence_length, idx

            inst_id = self.data[end - 1, 0]

            if self.config.global_stream:
                # Slide the pc_hist to the left by 1 if using current PC
                if self.config.use_current_pc:
                    pc_hist = self.data[start + 1:end + 1, 1]
                else:
                    pc_hist = self.data[start:end, 1]
                page_hist = self.data[start:end, 2]
                offset_hist = self.data[start:end, 3]
                hists.append(pc_hist)
                hists.append(page_hist)
                hists.append(offset_hist)

            if self.config.global_output:
                # Multi-label draws from the pages + offsets tensors while
                # Global stream only can work directly with the data tensor
                if self.config.multi_label:
                    if self.config.sequence_loss:
                        y_page = self.pages[pred_start + 1:pred_end + 1]
                        y_offset = self.offsets[pred_start + 1:pred_end + 1]
                    else:
                        y_page = self.pages[pred_end:pred_end + 1]
                        y_offset = self.offsets[pred_end:pred_end + 1]
                else:
                    if self.config.sequence_loss:
                        y_page = self.data[pred_start + 1:pred_end + 1, 2]
                        y_offset = self.data[pred_start + 1:pred_end + 1, 3]
                    else:
                        y_page = self.data[pred_end:pred_end + 1, 2]
                        y_offset = self.data[pred_end:pred_end + 1, 3]

            # PC Localized Input and Output
            cur_pc = self.data[idx, 1]
            end = self.data[idx, 4]
            start = end - self.config.sequence_length - self.config.prediction_depth

            if self.config.pc_localized:
                hist = tf.gather(self.data, self.pc_data[cur_pc, start:end + 1])
                page_hist = hist[:self.config.sequence_length, 2]
                offset_hist = hist[:self.config.sequence_length, 3]
                if self.config.use_current_pc:
                    pc_hist = hist[1:self.config.sequence_length + 1, 1]
                else:
                    pc_hist = hist[:self.config.sequence_length, 1]
                hists.append(pc_hist)
                hists.append(page_hist)
                hists.append(offset_hist)

            if not self.config.global_output:
                if self.config.multi_label:
                    page_hist = tf.gather(self.pages, self.pc_data[cur_pc, start:end + 1])
                    offset_hist = tf.gather(self.offsets, self.pc_data[cur_pc, start:end + 1])
                    if self.config.sequence_loss:
                        y_page = page_hist[1 + self.config.prediction_depth:, 2]
                        y_offset = offset_hist[1 + self.config.prediction_depth:, 3]
                    else:
                        y_page = page_hist[-1:, 2]
                        y_offset = offset_hist[-1:, 3]
                else:
                    if self.config.sequence_loss:
                        y_page = hist[1 + self.config.prediction_depth:, 2]
                        y_offset = hist[1 + self.config.prediction_depth:, 3]
                    else:
                        y_page = hist[-1:, 2]
                        y_offset = hist[-1:, 3]

            return inst_id, tf.concat(hists, axis=-1), y_page, y_offset

        # Closure for generating a reproducible random sequence
        epoch_size = self.config.steps_per_epoch * self.config.batch_size
        def random_closure(minval, maxval):
            def random(x):
                epoch = x // epoch_size
                step = x % epoch_size
                return tf.random.stateless_uniform((), minval=minval, maxval=maxval, seed=(epoch, step), dtype=tf.dtypes.int64)
            return random

        if online:
            # Sets of train and eval datasets for each epoch
            train_datasets = []
            eval_datasets = []
            # Exclude the first N values since they cannot fully be an input for Voyager
            cutoffs = [self.config.sequence_length + self.config.prediction_depth] + self.online_cutoffs

            # Include the rest of the data if it wasn't on a 50M boundary
            if self.online_cutoffs[-1] < len(self.data):
                self.online_cutoffs.append(len(self.data))

            # Stop before the last dataset since we don't need to train on it
            for i in range(start_phase, len(cutoffs) - 2):
                # Train on DATA[idx[i]] to DATA[idx[i + 1]]
                # Evaluate on DATA[idx[i + 1]] to DATA[idx[i + 2]]
                if i == start_phase:
                    # Resume from where you left off
                    train_datasets.append(
                        tf.data.Dataset.range(start_epoch * epoch_size + start_step * self.config.batch_size, self.config.num_epochs_online * epoch_size)
                            .map(random_closure(cutoffs[i], cutoffs[i + 1]))
                            .map(mapper)
                            .batch(self.config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                    )
                else:
                    train_datasets.append(
                        tf.data.Dataset.range(0, self.config.num_epochs_online * epoch_size)
                            .map(random_closure(cutoffs[i], cutoffs[i + 1]))
                            .map(mapper)
                            .batch(self.config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                    )

                eval_datasets.append(
                    tf.data.Dataset.range(cutoffs[i + 1], cutoffs[i + 2])
                        .map(mapper)
                        .batch(self.config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                )

            return train_datasets, eval_datasets

        # Regular 80-10-10 decomposition
        train_split, valid_split = self._split_idx()

        # Put together the datasets
        train_ds = (tf.data.Dataset
            .range(start_epoch * epoch_size + start_step * self.config.batch_size, self.config.num_epochs * epoch_size)
            .map(random_closure(self.config.sequence_length + self.config.prediction_depth, train_split))
            .map(mapper)
            .batch(self.config.batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        )

        valid_ds = (tf.data.Dataset.range(train_split, valid_split)
            .map(mapper)
            .batch(self.config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        )

        test_ds = (tf.data.Dataset.range(valid_split, len(self.data))
            .map(mapper)
            .batch(self.config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        )

        return train_ds, valid_ds, test_ds

    # Unmaps the page and offset
    def unmap(self, x, page, offset, sequence_length):
        unmapped_page = self.reverse_page_mapping[page]

        # DELTA LOCALIZED
        if isinstance(unmapped_page, str):
            prev_page = x[2 * sequence_length - 1]
            prev_offset = x[-1]
            unmapped_prev_page = self.reverse_page_mapping[prev_page]
            prev_addr = (unmapped_prev_page << self.config.offset_bits) + prev_offset
            delta = int(unmapped_page[1:])
            if unmapped_page[0] == '+':
                ret_addr = prev_addr + delta
            else:
                ret_addr = prev_addr - delta
        else:
            ret_addr = (unmapped_page << self.config.offset_bits) + offset

        return ret_addr << 6

def read_benchmark_trace(benchmark_path, config):
    '''
    Reads and processes the trace for a benchmark
    '''
    benchmark = BenchmarkTrace(config)

    if benchmark_path.endswith('.txt'):
        with open(benchmark_path, 'r') as f:
            benchmark.read_and_process_file(f)
    elif benchmark_path.endswith('.txt.xz'):
        with lzma.open(benchmark_path, mode='rt', encoding='utf-8') as f:
            benchmark.read_and_process_file(f)

    return benchmark
