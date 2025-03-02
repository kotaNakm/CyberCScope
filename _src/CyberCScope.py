""" Python implementation of CyberCScope """

import numpy as np
from scipy.special import digamma, gammaln
import scipy
from tqdm import trange, tqdm
import numba
import copy
from itertools import groupby
import time
import warnings
from math import lgamma

warnings.simplefilter("ignore")

FB: int

ZERO = 1.0e-8
REGIME_R = 3.0e-3
LAMBDA = 0.1
MAX_INI_r = 2
# MAX_INI_r = 1
SEED = 0
PDF_MIN = 1e-50 # fix


# FB = 2, 4, 8, 16, 24, 32



class Regime(object):
    def __init__(self):
        self.costM = np.inf
        self.costC = np.inf
        self.costT = np.inf

    def compute_costM(self, cnt, verbose: bool):
        if cnt == 0:
            return np.inf
        cost = 0
        tol_r = 2 ** (-FB)
        each_non_zero_counts = np.full(self.n_modes,2)
        non_zeros = np.sum((self.mn_factors[0] > tol_r).astype(int), axis=0)
        non_zeros[np.argmax(non_zeros)] = 0
        each_non_zero_counts[0] = non_zeros.sum()

        for mode_ in range(1, len(self.mn_factors)):
            non_zeros = np.sum((self.mn_factors[mode_] > tol_r).astype(int), axis=1)
            non_zeros[np.argmax(non_zeros)] = 0
            each_non_zero_counts[mode_] = non_zeros.sum()
    

        if verbose:
            print(f"each_non_zero_counts: {each_non_zero_counts}")
            print(f"n_dims: {self.n_dims}")
            print(f"FB: {FB}, TOL_R: {tol_r}")

        cost += log_s(cnt)
        
        self.n_dims_refined = copy.deepcopy(self.n_dims[:len(self.mn_factors)])
        self.n_dims_refined[1:][self.n_dims_refined[1:] < 2] = 2  # avoid log(0)
        cost += each_non_zero_counts[0] * (
            np.log2((self.k - 1) * self.n_dims_refined[0]) + FB
        )
        cost += log_s(each_non_zero_counts[0])
        for mode_ in range(1, len(self.mn_factors)):
            cost += each_non_zero_counts[mode_] * (
                np.log2(self.k * (self.n_dims_refined[mode_] - 1)) + FB
            )
            cost += log_s(each_non_zero_counts[mode_])
        for mode_ in range(len(self.gm_factors)):
            cost += self.k * 2 * FB
        self.costM = cost
        return cost

    def compute_costC(self, X):
        if len(X) == 0:
            return 0
        self.mn_factors = _normalize_factors(self.mn_factors)
        
        costC = _compute_costC(X.to_numpy(), self.mn_factors, self.gm_factors)
        
        # n_all_categorical_idxs = len(self.mn_factors)
        # gm_cost = _compute_gm_llh(X.to_numpy()[:,n_all_categorical_idxs:], self.gm_factors)
        return costC


    def compute_factors(self):
        
        self.mn_factors[0] = (
            (self.counterM[0][:] + self.prev_terms[0][:]).T
            / (self.counterA + self.l * self.alpha)
        ).T

        for mode_ in range(1, len(self.categorical_idxs)+1):
            self.mn_factors[mode_] = (self.counterM[mode_] + self.prev_terms[mode_]) / (
                self.counterK + self.l * self.betas[mode_ - 1]
            )
        self.mn_factors = _normalize_factors(self.mn_factors)

        for mode_, _ in enumerate(self.continuous_idxs):
            shapes, scales = get_params(self.suffstatsMc[mode_])
            self.gm_factors[mode_][0] = shapes
            self.gm_factors[mode_][1] = scales

        return self.mn_factors, self.gm_factors

    # def compute_factors(self):
    #     self.factors[0] = (
    #         (self.counterM[0][:] + self.prev_terms[0][:]).T
    #         / (self.counterA + self.l * self.alpha)
    #     ).T

    #     for mode_ in range(1, self.n_modes):
    #         self.factors[mode_] = (self.counterM[mode_] + self.prev_terms[mode_]) / (
    #             self.counterK + self.l * self.betas[mode_ - 1]
    #         )
    #     self.factors = _normalize_factors(self.factors)


class CyberCScope(object):
    def __init__(
        self,
        tensor,
        k: int,
        width: int,
        init_len: int,
        outputdir: str,
        time_idx: str,
        categorical_idxs:list,
        continuous_idxs:list,
        args: object,
        verbose: bool,
        keep_best_factors: bool = True,
        early_stoppping: bool = False,
    ):
        # initialze
        self.k = k  # # of topics/components
        self.width = width
        self.init_len = init_len

        self.time_idx = time_idx
        self.categorical_idxs = categorical_idxs
        self.continuous_idxs = continuous_idxs
        self.n_dims = np.full(1+len(self.categorical_idxs) + len(self.continuous_idxs),2)
        self.n_dims[0] = width
        for i, cate in enumerate(self.categorical_idxs):
            self.n_dims[1+i] = tensor[cate].max() + 1
        self.n_dims = self.n_dims.astype(int)
        self.prev_dims = copy.deepcopy(self.n_dims)
        self.prev_dims[len(self.categorical_idxs)+1:] = 10
        self.n_modes = len(self.n_dims)
        

        print("##tensor dimensions##")
        print(self.n_dims)

        self.cur_n = 0

        self.init_params(args.alpha, args.beta)  # init alpha, beta and factors
        self.regimes = []

        self.keep_best_factors = keep_best_factors
        self.early_stoppping = early_stoppping

        # for anormaly detection
        self.anomaly = args.anomaly
        if self.anomaly:
            self.anomaly_scores = []

        # for visualization
        self.outputdir = outputdir
        self.verbose: bool = verbose
        self.sampling_log_likelihoods = []

        # for update parameters
        self.max_alpha = 100
        self.max_beta = 100

    def init_params(self, alpha, beta):
        """Initialize alpha, beta and factors"""
        self.alpha = np.full(self.n_dims[0], alpha)
        self.betas = [np.full(self.k, beta) for _ in range(self.n_modes - 1)]
        self.mn_factors = [np.full((d, self.k), 1, dtype=float) for d in self.n_dims[:len(self.categorical_idxs)+1]]
        self.gm_factors = [np.full((2, self.k), 1, dtype=float) for _ in self.continuous_idxs]

    def init_status(self, tensor):
        """Initialize Counters for current tensor"""
        """
            suffstats[0] = a
            suffstats[1] = b
            suffstats[2] = c
            suffstats[3] = d
            suffstats[4] = e
            suffstats[5] = N
            suffstats[6] = S
            suffstats[7] = lnS
            suffstats[8] = mu
            suffstats[9] = var
        """  

        self.counterK = np.zeros(self.k, dtype=int)
        self.counterA = np.zeros(self.n_dims[0], dtype=int)
        self.counterM = [np.zeros((self.n_dims[d], self.k), dtype=int) for d, _ in enumerate([self.time_idx]+self.categorical_idxs)]
        self.suffstatsMc = []
        for d, _ in enumerate(self.continuous_idxs):
            self.suffstatsMc.append(np.ones((10, self.k), dtype=float))
            self.suffstatsMc[d][5:,:] = 1e-10

        self.n_events = len(tensor)
        self.assignment = np.full(self.n_events, -1, dtype=int)

        Asum = tensor.groupby(self.keys[0]).size()
        self.counterA[Asum.index] = Asum.values

    def init_infer(
        self,
        tensor_train,
        n_iter=10,
        init_l: bool = True,
        return_inference_time=False,
        calc_computational_cost=False,
    ):
        """Initialize model parameters i.e, training process
        1. batch estimation for each subtensor
        2. Initialize model parameters employing subtensors given by 1.
        """
        self.time_idx = list(tensor_train.columns)[0]
        self.keys = list(tensor_train.keys())

        self.l = int(tensor_train[self.time_idx].max() // self.width)
        # the oldest regime index is assigned 0, l is the newest index

        self.prev_distributions = [
            np.full((self.l, d, self.k), 1, dtype=float) for i, d in enumerate(self.prev_dims)
        ]
        # self.prev_suffstats = [
        #     np.full((self.l, 10, self.k), 1, dtype=float) for d, _ in enumerate(self.continuous_idxs)
        # ]

        # self.l = len(tensor_train) // self.width
        print(f"l: {self.l}")

        if self.l < 1:
            print("Data[:init_len] does not have enough data for initialization")
            print("Inlier records in data[:init_len] must be longer than width")
            exit()

        # 1. process for subtensors that given by train tensor devided by l
        best_llh_in_L = -np.inf
        best_l = 0

        for ini in range(self.l):
            tensor = tensor_train[
                (tensor_train[self.time_idx] >= ini * self.width)
                & (tensor_train[self.time_idx] < (ini + 1) * self.width)
            ]
            tensor.loc[:, self.time_idx] -= ini * self.width
            print(tensor)
            cnt = len(tensor)
            self.init_status(tensor)

            if self.verbose:
                print("Gibbs Sampling")
            self.each_samp_llh = []
            best_llh = -np.inf
            start_inference_time = time.process_time()
            for iter_ in range(n_iter):
                self.assignment = self.sample_topic(tensor.to_numpy())
                # compute log likelihood
                llh = self.log_likelihood()
                if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                    print(f"llh_{iter_+1}: {llh}")
                self.each_samp_llh.append(llh)

                if (self.early_stoppping) and (
                    self.each_samp_llh[-2] > self.each_samp_llh[-1]
                ):
                    break

                if (self.keep_best_factors) and (llh > best_llh):
                    best_counterM = copy.deepcopy(self.counterM)
                    best_suffstatsMc = copy.deepcopy(self.suffstatsMc)
                    best_counterK = copy.deepcopy(self.counterK)
                    best_llh = llh

            inference_time = time.process_time() - start_inference_time
            if calc_computational_cost:
                return 0, inference_time
            # print(self.assignment)
            if self.keep_best_factors:
                self.counterM = best_counterM
                self.counterK = best_counterK
                self.suffstatsMc = best_suffstatsMc
                print(f"best llh: {self.log_likelihood()}")

            self.sampling_log_likelihoods.append(self.each_samp_llh)
            self.compute_factors_batch()
            self.update_prev_dist_init(cnt, ini)

            if best_llh_in_L < best_llh:  # higher is better
                best_l = ini

        # # choice only best prev distribution
        initial_prev_dist = [
            self.prev_distributions[m][best_l] for m in range(self.n_modes)
        ]
        self.prev_distributions = [
            np.full((self.l, d, self.k), 0, dtype=float) for d in self.prev_dims
        ]
        for m in range(self.n_modes):
            self.prev_distributions[m][0] = initial_prev_dist[m]
        self.vscost_log = []

        # 2. initialize model parameters
        if return_inference_time:
            return self.model_initialization(tensor_train, n_iter), inference_time
        else:
            return self.model_initialization(
                tensor_train,
                n_iter,
            )

    def model_initialization(
        self,
        tensor_all,
        n_iter,
    ):
        """
        * infer tracking factors using factors derived by batch estimation (init_infer)
        * determine optimal l
        """

        def segment_len(rgm_nums, alloc):
            """
            return a half of max segment length
            return a max segment length
            """
            max_L = 0
            for num in rgm_nums:
                # print(num)
                tmp = np.where(alloc == num, 1, -1)
                dst = [
                    sum(1 for e in it) for _, it in groupby(tmp, key=lambda x: x > 0)
                ]
                # print(dst)
                if tmp[0] > 0:
                    max_ = np.max(dst[::2])
                else:
                    max_ = np.max(dst[1::2])

                if max_ > max_L:
                    max_L = max_
            return int(max_L)

        ini_regimes = []
        self.regime_hist=[]

        regimes_cost = {}
        for ini in range(self.l):
            # get partial tensor as current tensor
            tensor = tensor_all[
                (tensor_all[self.time_idx] >= ini * self.width)
                & (tensor_all[self.time_idx] < (ini + 1) * self.width)
            ]
            cnt = len(tensor)
            tensor.loc[:, self.time_idx] -= ini * self.width

            self.init_status(tensor)

            if self.verbose:
                print("Online Gibbs Sampling")
            print(f"# of events: {cnt}")
            self.each_samp_llh = []
            best_llh = -np.inf
            for iter_ in range(n_iter):
                self.assignment, self.prev_terms = self.sample_topic_online(
                    tensor.to_numpy()
                )

                # compute log likelihood
                llh = self.log_likelihood()
                if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                    print(f"llh_{iter_+1}: {llh}")
                self.each_samp_llh.append(llh)

                if (self.early_stoppping) and (
                    self.each_samp_llh[-2] > self.each_samp_llh[-1]
                ):
                    break

                if (self.keep_best_factors) and (llh > best_llh):
                    best_counterM = copy.deepcopy(self.counterM)
                    best_counterK = copy.deepcopy(self.counterK)
                    best_suffstatsMc = copy.deepcopy(self.suffstatsMc)    
                    best_llh = llh

            if self.keep_best_factors:
                self.counterM = best_counterM
                self.counterK = best_counterK
                self.suffstatsMc = best_suffstatsMc
                print(f"best llh: {best_llh}")

            self.sampling_log_likelihoods.append(self.each_samp_llh)
            self.compute_factors()
            self.update_prev_dist_init(cnt, ini)

            # find opt regime
            # choose the regime which has most lowest coding cost
            tmp_rgm = Regime()
            tmp_rgm = self.regime_initialize(tmp_rgm)

            tmp_costC = tmp_rgm.compute_costC(tensor)
            tmp_costM = tmp_rgm.compute_costM(len(tensor), verbose=self.verbose)

            ini_regimes.append(tmp_rgm)
            regimes_cost[ini] = tmp_costC + tmp_costM
            print(f"initialized costC for train segment #{ini}: {tmp_costC}")
            print(f"initialized costM for train segment #{ini}: {tmp_costM}")

        if self.anomaly:
            self.aggregate_initials(ini_regimes) # ensure the inialize set in a single regime
            self.l = 1
            regime_assignments = [[0, 0]]
            self.regimes.append(self.all_comp_regime)

            # update prev dist by all_comp regime as a regime
            for mode_ in range(self.n_modes):
                if mode_ < 1+len(self.categorical_idxs):
                    self.prev_distributions[mode_][0, :, :] = copy.deepcopy(
                        self.all_comp_regime.mn_factors[mode_]
                    )
                else:
                    updated_suffstats = update_stats(self.suffstatsMc[mode_-(1+len(self.categorical_idxs))])[0]
                    self.prev_distributions[mode_][0, :, :] = copy.deepcopy(
                    updated_suffstats
                    )

            self.prev_distributions = [
                copy.deepcopy(prev_dist_mode[:1])
                for prev_dist_mode in self.prev_distributions
            ]
            self.cur_n = self.init_len
            self.prev_rgm_id = 0
            self.n_segment = 0
            return regime_assignments

        # determine initial regime set and regime assignments by computing total cost for train tensor
        opt_rgm_nums, opt_alloc = self.determine_initial_regimeset_assignments(
            regimes_cost, tensor_all, ini_regimes
        )

        # add optimal regimes to regime set for stream process
        for i, r_id in enumerate(opt_rgm_nums):
            self.regimes.append(ini_regimes[r_id])
            opt_alloc[opt_alloc == r_id] = i

        self.prev_rgm_id = opt_alloc[-1]

        # determine optimal l following segment length and update prev_dist
        L = segment_len(range(len(opt_rgm_nums)), opt_alloc)
        self.l = L
        print(f"dependencies length L:{L}")
        self.prev_distributions = [
            copy.deepcopy(prev_dist_mode[:L])
            for prev_dist_mode in self.prev_distributions
        ]

        regime_assignments = [[0, 0]]

        return regime_assignments

    def determine_initial_regimeset_assignments(
        self, regimes_cost, tensor_all, ini_regimes
    ):
        cost_sorted = sorted(
            regimes_cost.items(), key=lambda x: x[1]
        )  # order of regime numbers which has smaller cost
        min_ = np.inf
        opt_alloc = np.full(len(ini_regimes), 0)
        opt_rgm_nums = [
            0,
        ]
        for max_r_num in range(1, MAX_INI_r + 1):
            candidate_rgm_nums = [num[0] for num in cost_sorted[:max_r_num]]
            print(f"candidate regimes for initial regimes:{candidate_rgm_nums}")
            cost, alloc = self.compute_total_cost_batch(
                tensor_all, ini_regimes, candidate_rgm_nums
            )

            if cost < min_:
                opt_alloc = alloc
                opt_rgm_nums = np.unique(alloc)
                min_ = cost
            else:
                print("break")
                print(f"top samllest {max_r_num-1} regimes")
                print(f"MAX_INI_r: {MAX_INI_r}")
                break
        # count segment
        self.n_segment = np.count_nonzero(
            np.array(opt_alloc[1:]) - np.array(opt_alloc[:-1])
        )

        print(f"initial regimes: {opt_rgm_nums}")
        print(f"cost and assignment: {min_} & {opt_alloc}")

        return opt_rgm_nums, opt_alloc

    def compute_total_cost_batch(self, tensor_all, ini_regimes, cadidate_rgm_nums):
        """
        compute total cost employed ini_regimes[candidate_rgm_nums] and assign them
        Return:
        cost: total cost computed by ini_regimes[candidate_rgm_nums]
        alloc: candidate regime assignments
        """

        alloc = np.full(self.l, -1)
        cost = 0
        # calc total coding cost and assign
        for ini in range(self.l):
            min_ = np.inf
            target_tensor = tensor_all[
                (tensor_all[self.time_idx] >= ini * self.width)
                & (tensor_all[self.time_idx] < (ini + 1) * self.width)
            ]
            target_tensor.loc[:, self.time_idx] -= ini * self.width
            for r in cadidate_rgm_nums:
                r_cost = ini_regimes[r].compute_costC(target_tensor)
                if min_ > r_cost:
                    alloc[ini] = r
                    min_ = r_cost
            cost += min_
        # add model cost to cost
        for r_id in np.unique(alloc):
            cost += ini_regimes[r_id].costM
        return cost, alloc

    def infer_online(self, tensor, alpha, beta, n_iter=10):
        # whether apply tuned parameter in initialization
        if True:
            self.init_params(alpha, beta)

        self.init_status(tensor)

        self.cur_n += self.width
        if self.verbose:
            print("Online Gibbs Sampling")
        print(f"# of events: {len(tensor)}")

        self.each_samp_llh = []
        best_llh = -np.inf

        start_time = time.process_time()
        for iter_ in range(n_iter):
            self.assignment, self.prev_terms = self.sample_topic_online(
                tensor.to_numpy()
            )
            # compute log likelihood
            llh = self.log_likelihood()
            if self.verbose or iter_ == 0 or iter_ == (n_iter - 1):
                print(f"llh_{iter_+1}: {llh}")
            self.each_samp_llh.append(llh)

            if (self.early_stoppping) and (
                self.each_samp_llh[-2] > self.each_samp_llh[-1]
            ):
                break
            if (self.keep_best_factors) and (llh > best_llh):
                best_counterM = copy.deepcopy(self.counterM)
                best_counterK = copy.deepcopy(self.counterK)
                best_suffstatsMc = copy.deepcopy(self.suffstatsMc)    
                best_llh = llh
        decomp_time = time.process_time() - start_time
        print(f"decomp_time:{decomp_time}")

        start_time = time.process_time()
        if self.keep_best_factors:
            self.counterM = best_counterM
            self.counterK = best_counterK
            self.suffstatsMc = best_suffstatsMc
            print(f"best llh: {self.log_likelihood()}")

        self.sampling_log_likelihoods.append(self.each_samp_llh)
        self.cnt = len(tensor)
        self.compute_factors()
        self.update_prev_dist(self.cnt, self.l)

        shift_id = self.model_compressinon(self.cur_n, tensor)

        compress_time = time.process_time() - start_time
        print(f"compress_time:{compress_time}")

        return shift_id

    def sample_topic(self, X):
        return _gibbs_sampling(
            X,
            self.assignment,
            self.counterM,
            self.counterK,
            self.counterA,
            self.alpha,
            self.betas,
            self.suffstatsMc,
            self.k,
            self.n_dims,
        )

    def sample_topic_online(self, X):
        return _gibbs_sampling_online(
            X,
            self.assignment,
            self.counterM,
            self.counterK,
            self.counterA,
            self.alpha,
            self.betas,
            self.suffstatsMc,
            self.k,
            self.prev_distributions,
            self.l,
            self.prev_dims,
        )

    def log_likelihood(
        self,
    ):
        llh = 0
        for i in range(self.n_dims[0]):
            llh += log_multi_beta(self.counterM[0][i, :] + self.alpha[i])
            llh -= log_multi_beta(self.alpha[i], self.k)  # ??

        for mode_, _ in enumerate(self.categorical_idxs):
            for i in range(self.k):
                llh += log_multi_beta(
                    self.counterM[mode_ + 1][:, i] + self.betas[mode_][i]
                )
                llh -= log_multi_beta(self.betas[mode_][i], self.n_dims[0])
        for mode_, _ in enumerate(self.continuous_idxs):
            suffstats = self.suffstatsMc[mode_]
            shapes, scales = get_params(suffstats)
            # print(f"mode:{mode_}")
            # print(f"shapes:{shapes}\nscales:{scales}")
            for i in range(self.k): 
                N = suffstats[5,i]
                S = suffstats[6,i]
                lnS = suffstats[7,i]
                llh += llh_gamma(N, S, lnS, shapes[i], scales[i])
        return llh
    

    def update_prev_dist(self, cnt, l):
        """
        push and pop prev_dist queue
        """

        if cnt:
            for mode_ in range(1+len(self.categorical_idxs)):
                self.prev_distributions[mode_][:-1] = copy.deepcopy(
                    self.prev_distributions[mode_][1:]
                )
                self.prev_distributions[mode_][-1, :, :] = copy.deepcopy(
                    self.mn_factors[mode_]
                )

            for mode_ in range(len(self.continuous_idxs)):
                updated_suffstats = update_stats(self.suffstatsMc[mode_])[0]
                self.prev_distributions[1+len(self.categorical_idxs)+mode_][-1, :, :] = copy.deepcopy(
                    updated_suffstats
                )


    def update_prev_dist_init(self, cnt, l):
        """add estimated factors to prev_dist[l]"""
        if cnt:
            for mode_ in range(1+len(self.categorical_idxs)):
                self.prev_distributions[mode_][l, :, :] = copy.deepcopy(
                    self.mn_factors[mode_]
                )
            for mode_ in range(len(self.continuous_idxs)):
                updated_suffstats = update_stats(self.suffstatsMc[mode_])[0]
                self.prev_distributions[1+len(self.categorical_idxs)+mode_][l, :, :] = copy.deepcopy(
                    updated_suffstats
                )
        # for visualize
        if False:
            print(f"prev distributions for {l}")
            for mode_ in range(self.n_modes):
                print(self.prev_distributions[mode_])

    def compute_factors_batch(self):
        for i in range(self.n_dims[0]):
            self.mn_factors[0][i, :] = (self.counterM[0][i, :] + self.alpha[i]) / (
                self.counterA[i] + self.alpha[i] * self.k
            )

        for mode_ in range(1, len(self.categorical_idxs)+1):
            for i in range(self.n_dims[mode_]):
                self.mn_factors[mode_][i, :] = (
                    self.counterM[mode_][i, :] + self.betas[mode_ - 1]
                ) / (self.counterK + self.betas[mode_ - 1] * self.n_dims[mode_])
        self.mn_factors = _normalize_factors(self.mn_factors)

        for mode_, _ in enumerate(self.continuous_idxs):
            shapes, scales = get_params(self.suffstatsMc[mode_])
            self.gm_factors[mode_][0] = shapes
            self.gm_factors[mode_][1] = scales

    def compute_factors(self):
        self.mn_factors[0] = (
            (self.counterM[0][:] + self.prev_terms[0][:]).T
            / (self.counterA + self.l * self.alpha)
        ).T

        for mode_ in range(1, len(self.categorical_idxs)+1):
            self.mn_factors[mode_] = (self.counterM[mode_] + self.prev_terms[mode_]) / (
                self.counterK + self.l * self.betas[mode_ - 1]
            )
        self.mn_factors = _normalize_factors(self.mn_factors)

        for mode_, _ in enumerate(self.continuous_idxs):
            shapes, scales = get_params(self.suffstatsMc[mode_])
            self.gm_factors[mode_][0] = shapes
            self.gm_factors[mode_][1] = scales

    def aggregate_initials(self, init_regimes):
        tmp_rgm = Regime()
        tmp_rgm = self.regime_initialize(tmp_rgm)

        all_count = 0
        for ini_rgm in init_regimes:
            tmp_rgm.counterM += copy.deepcopy(ini_rgm.counterM)
            tmp_rgm.counterK += copy.deepcopy(ini_rgm.counterK)
            tmp_rgm.counterA += copy.deepcopy(ini_rgm.counterA)
            all_count += ini_rgm.counterK.sum()

        tmp_rgm.counterM = [m.astype(float) for m in ini_rgm.counterM]
        tmp_rgm.counterK = ini_rgm.counterK.astype(float)
        tmp_rgm.counterA = ini_rgm.counterA.astype(float)
        tmp_rgm.suffstatsMc = ini_rgm.suffstatsMc
        tmp_rgm.compute_factors()


        self.all_comp_regime = tmp_rgm
        self.all_count = all_count

    def model_compressinon(self, cur_n, X):
        shift_id = False
        self.n_regime = len(self.regimes)
        prev_rgm = self.regimes[self.prev_rgm_id]
        candidate_rgm = Regime()
        candidate_rgm = self.regime_initialize(candidate_rgm)

        ## compute_costM
        costM = candidate_rgm.compute_costM(self.cnt, verbose=self.verbose)
        costM += log_s(self.n_regime + 1) - log_s(self.n_regime)
        costM += (
            (log_s(self.n_segment + 1) - log_s(self.n_segment))
            + log_s(self.cnt)
            + np.log2(self.n_regime)
        )
        costC = candidate_rgm.compute_costC(X)
        cost_1 = costC + costM

        print(f"cand costC: {costC}")
        print(f"cand costM: {costM}")
        print(f"cand costT: {cost_1}")

        cost_0 = prev_rgm.compute_costC(X)
        print(f"prev costT: {cost_0}")

        self.vscost_log.append([cost_0, cost_1, cost_1 - cost_0, costC, costM])

        print("=========================================")
        print(f"Previous vs Candidate")
        print(f"{cost_0} vs {cost_1}")
        print("=========================================")
        print(f"diff::{cost_1 - cost_0}")

        if cost_0 < cost_1:  # stay on previous regime
            print("STAY")
            prev_rgm = self.regimes[self.prev_rgm_id]
            self.regime_update(self.prev_rgm_id)
        
        else:  # shift to any regime
            self.n_segment += 1
            shift_id = len(self.regimes)  # index + 1
            min_ = cost_1 + REGIME_R * cost_1
            add_flag = True

            # regime comparison
            for rgm_id, rgm in enumerate(self.regimes):
                if rgm_id == self.prev_rgm_id:
                    continue
                else:
                    rgm_costC = rgm.compute_costC(X)
                    rgm_costM = (
                        (log_s(self.n_segment + 1) - log_s(self.n_segment))
                        + log_s(self.cnt)
                        + np.log2(self.n_regime)
                    )
                    cost_0 = rgm_costC + rgm_costM
                    if cost_0 < min_:
                        shift_id = rgm_id
                        add_flag = False
                        min_ = cost_0

            print(f"SHIFT at {cur_n}")
            print(f"{self.prev_rgm_id}===>>>{shift_id}")

            if add_flag:  # add candidate regime to regime set
                self.regimes.append(candidate_rgm)
                self.prev_rgm_id = shift_id

            else:  # use existing regime
                self.regime_update(shift_id)
                shift_rgm = self.regimes[shift_id]
                self.alpha = copy.deepcopy(shift_rgm.alpha)
                self.betas = copy.deepcopy(shift_rgm.betas)
                self.prev_rgm_id = shift_id

        # For anormaly detection
        if self.anomaly:
            self.regime_hist.append(copy.deepcopy(self.prev_rgm_id))
            current_count = self.n_events
            self.all_count += current_count
            
            rgm_ids, counts = np.unique(self.regime_hist, return_counts=True)
            longest_rgm_id = rgm_ids[np.argmax(counts)]

            observed_cost = costC
            expected_cost = self.regimes[longest_rgm_id].compute_costC(X)

            a_score = self.calc_anomaly_score(observed_cost, expected_cost, len(X))
            self.anomaly_scores.append(a_score)
            print("===Anomaly score===")
            print(a_score)

        return shift_id

    def calc_anomaly_score(self, observed_cost, expected_cost, len_X):
        if expected_cost == 0:
            return 0
        else:
            a_scroe = expected_cost / len_X

        return a_scroe

    def regime_initialize(self, regime_instance):
        regime_instance.k = self.k
        regime_instance.n_dims = self.n_dims
        regime_instance.prev_dims = self.prev_dims
        regime_instance.n_modes = self.n_modes
        regime_instance.l = self.l
        regime_instance.time_idx = self.time_idx
        regime_instance.categorical_idxs = self.categorical_idxs
        regime_instance.continuous_idxs = self.continuous_idxs
        regime_instance.mn_factors = copy.deepcopy(self.mn_factors)
        regime_instance.gm_factors = copy.deepcopy(self.gm_factors)
        regime_instance.counterM = copy.deepcopy(self.counterM)
        regime_instance.counterK = copy.deepcopy(self.counterK)
        regime_instance.counterA = copy.deepcopy(self.counterA)
        regime_instance.suffstatsMc = copy.deepcopy(self.suffstatsMc)
        regime_instance.alpha = self.alpha
        regime_instance.betas = self.betas
        regime_instance.prev_terms = self.prev_terms

        return regime_instance

    def regime_update(self, rgm_id):
        regime = self.regimes[rgm_id]

        regime.counterK = regime.counterK.astype(float)
        regime.counterK += np.round(self.counterK * LAMBDA)
        regime.counterA = regime.counterA.astype(float)
        regime.counterA += np.round(self.counterA * LAMBDA)

        for mode_ in range(1+len(self.categorical_idxs)):
            regime.counterM[mode_] = regime.counterM[mode_].astype(float)
            regime.counterM[mode_] += np.round(self.counterM[mode_] * LAMBDA)
        
        # TODO
        for mode_ in range(len(self.continuous_idxs)):
            regime.suffstatsMc[mode_] = regime.suffstatsMc[mode_].astype(float)
            regime.suffstatsMc[mode_] += np.round(self.suffstatsMc[mode_] * LAMBDA)

        regime.mn_factors,regime.gm_factors = regime.compute_factors()

    def rgm_update_fin(self):
        if self.prev_rgm_id == (len(self.regimes) - 1):
            rgm = self.regimes[self.prev_rgm_id]
            rgm.alpha = copy.deepcopy(self.alpha)
            rgm.betas = copy.deepcopy(self.betas)
            rgm.mn_factors = copy.deepcopy(self.mn_factors)
            rgm.gm_factors = copy.deepcopy(self.gm_factors)

    def save(self, outdir):
        """
        Save all of parameters for CyberCScope
        """
        if len(self.vscost_log) > 0:
            np.savetxt(outdir + "vs_cost.txt", self.vscost_log[-1])
        np.savetxt(outdir + "llh.txt", self.each_samp_llh)
        np.savetxt(outdir + "alpha.txt", self.alpha)
        np.savetxt(outdir + "betas.txt", self.betas)
        for i, M in enumerate(self.mn_factors):
            np.savetxt(outdir + "/mn_factor_{}.txt".format(i), M)
        for i, M in enumerate(self.gm_factors):
            np.savetxt(outdir + "/gm_factor_{}.txt".format(i), M)
        for i, M in enumerate(self.regimes[self.prev_rgm_id].mn_factors):
            np.savetxt(outdir + "/c_regime_mn_factor_{}.txt".format(i), M)
        for i, M in enumerate(self.regimes[self.prev_rgm_id].gm_factors):
            np.savetxt(outdir + "/c_regime_gm_factor_{}.txt".format(i), M)


@numba.jit(nopython=True)
def _gibbs_sampling(X, Z, counterM, counterK, counterA, alpha, betas, suffstatsMc, k, n_dims):
    np.random.seed(SEED)
    """
    X: event tensor
    Z: topic/component assignments of the previous iteration
    """
    n_modes = X.shape[1]
    n_all_categorical_modes = len(counterM)
    n_continuous_modes = len(suffstatsMc)

    for e, x in enumerate(X):
        # for each non-zero event entry,
        # assign latent component, z
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            for mode_ in range(n_all_categorical_modes):
                counterM[mode_][int(x[mode_]), pre_topic] -= 1
            for mode_ in range(n_continuous_modes):
                suffstatsMc[mode_] = decrement_Mc(x[n_all_categorical_modes+mode_], pre_topic, suffstatsMc[mode_])

        """ compute posterior distribution """
        posts = np.full(k, 1.0, dtype=np.float64)
        posts *= counterM[0][int(x[0])] + alpha[int(x[0])]  # return (k,) vector
        posts /= counterA[int(x[0])] + alpha[int(x[0])] * k
        for j in range(1,n_all_categorical_modes):
            posts *= counterM[j][int(x[j])] + betas[j-1]
            posts /= counterK + betas[j-1] * n_dims[j]
        for j in range(n_continuous_modes):
            # print(f"continous mode:{j}")
            # print(pdf_k(x[n_all_categorical_modes+j], suffstatsMc[j]))
            posts *= np.clip(pdf_k(x[n_all_categorical_modes+j], suffstatsMc[j]),PDF_MIN, None)
        posts = posts / posts.sum()

        try:
            # print("final") #TODO
            # print(posts) #TODO
            new_topic = draw_one(posts)
        except:
            print("cannot calc assignment posterior:")
            return

        Z[e] = new_topic
        counterK[new_topic] += 1
        for mode_ in range(n_all_categorical_modes):
            counterM[mode_][int(x[mode_]), new_topic] += 1
        for mode_ in range(n_continuous_modes):
            suffstatsMc[mode_] = increment_Mc(x[n_all_categorical_modes+mode_], new_topic, suffstatsMc[mode_])
    return Z


@numba.jit(nopython=True)
def _gibbs_sampling_online(
    X, Z, counterM, counterK, counterA, alpha, betas, suffstatsMc, k, prev_distributions, l, prev_dims
):
    np.random.seed(SEED)

    n_modes = X.shape[1]
    n_all_categorical_modes = len(counterM)
    n_continuous_modes = len(suffstatsMc)

    # ready for (hypa * prev)
    prev_terms = []
    for mode_ in range(n_modes):
        d = prev_dims[mode_]
        prev_terms.append(np.zeros((d, k)))
    for p in range(l):
        for i in range(prev_dims[0]):
            prev_terms[0][i, :] += alpha[i] * prev_distributions[0][p, i, :]
        for mode_ in range(1, n_all_categorical_modes):
            for i in range(k):
                prev_terms[mode_][:, i] += (
                    betas[mode_ - 1][i] * prev_distributions[mode_][p][:, i]
                )
        for mode_ in range(n_continuous_modes):
            for i in range(k):
                prev_terms[mode_+n_all_categorical_modes][:, i] += (
                    prev_distributions[mode_+n_all_categorical_modes][p][:, i]/l
                )
            # print(suffstatsMc)
            suffstatsMc[mode_][8,:] = prev_terms[mode_+n_all_categorical_modes][8,:]
            suffstatsMc[mode_][9,:] = prev_terms[mode_+n_all_categorical_modes][9,:]

    for e, x in enumerate(X):
        # for each non-zero event entry,
        # assign latent topic/component, z
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            for mode_ in range(n_all_categorical_modes):
                counterM[mode_][int(x[mode_]), pre_topic] -= 1
            for mode_ in range(n_continuous_modes):
                suffstatsMc[mode_] = decrement_Mc(x[n_all_categorical_modes+mode_], pre_topic, suffstatsMc[mode_])

        """ compute posterior distribution """
        posts = np.full(k, 1.0, dtype=np.float64)
        posts *= counterM[0][int(x[0])] + prev_terms[0][int(x[0])]  # return (k,) vector
        posts /= counterA[int(x[0])] + l * alpha[int(x[0])]
        for j in range(1, n_all_categorical_modes):
            posts *= counterM[j][int(x[j])] + prev_terms[j][int(x[j])]
            posts /= counterK + l * betas[j - 1]
        for j in range(n_continuous_modes):
            posts *= np.clip(pdf_k(x[n_all_categorical_modes+j], suffstatsMc[j]),PDF_MIN, None)
        posts = posts / posts.sum()
        try:
            new_topic = draw_one(posts)
        except:
            print("cannot calc assignment posterior:")
            print(posts)
            return

        Z[e] = new_topic
        counterK[new_topic] += 1
        for mode_ in range(n_all_categorical_modes):
            counterM[mode_][int(x[mode_]), new_topic] += 1
        for mode_ in range(n_continuous_modes):
            suffstatsMc[mode_] = increment_Mc(x[n_all_categorical_modes+mode_], new_topic, suffstatsMc[mode_])

    return Z, prev_terms


def log_s(x):
    if x == 0:
        return 0
    return 2.0 * np.log2(x) + 1


@numba.jit(nopython=True)
def _compute_costC(X, mn_factors, gm_factors):
    n_all_categorical_idxs = len(mn_factors)
    k = mn_factors[0].shape[1]
    all_L = 0
    for x in X:
        val_ = ZERO
        for r in range(k):
            rval_ = 0
            for att_idx, factor in zip(x[:n_all_categorical_idxs], mn_factors):
                rval_ += np.log(factor[int(att_idx), r] + ZERO)
            for att_value, factor in zip(x[n_all_categorical_idxs:], gm_factors):
                shape, scale = factor[:,r]
                # rval_ += np.log(gamma_pdf(att_value+ZERO, shape, scale))
                rval_ += llh_gamma(1, att_value+ZERO, np.log(att_value+ZERO), shape, scale)
            val_ = rval_ if r == 0 else np.logaddexp(val_, rval_)
        all_L -= val_
    # transform base
    return all_L / np.log(2.0)



# calc log(sum(exp(x))) in numerically stable way
@numba.jit(nopython=True)
def logsumexp(x, y, k):
    if k == 0:
        return y
    if x == y:
        return x + 0.69314718055  # ln(2)
    vmax = x if x > y else y
    vmin = y if x > y else x
    if vmax > (vmin + 50):
        return vmax
    else:
        return vmax + np.log1p(np.exp(vmin - vmax))

# @numba.jit(nopython=True)
def _normalize_factors(factors):
    """
    refine each components
    factors[0]: item-wise topic/component distribution
    factors[1:]: topic/component-wise item distribution
    """

    for d in range(factors[0].shape[0]):
        sum_ = np.sum(factors[0], axis=1) + ZERO
        factors[0] = (factors[0].T / sum_).T

    for ind, factor in enumerate(factors[1:]):
        sum_ = np.sum(factor, axis=0) + ZERO
        factors[ind + 1] /= sum_
    return factors


@numba.jit(nopython=True)
def draw_one(posts):
    residual = np.random.uniform(0, np.sum(posts))
    return_sample = 0
    for sample, prob in enumerate(posts):
        residual -= prob
        if residual < 0.0:
            return_sample = sample
            break
    return return_sample

@numba.jit(nopython=True)
def decrement_Mc(x, z, suffstats,eps=1e-10):      
    """
        suffstats[0,:] = ln_a
        suffstats[1,:] = b
        suffstats[2,:] = c
        suffstats[3,:] = d
        suffstats[4,:] = e
        suffstats[5,:] = N
        suffstats[6,:] = S
        suffstats[7,:] = lnS
        suffstats[8,:] = mu
        suffstats[9,:] = var
    """  
    x += eps
    mu_p = suffstats[8,z]
    suffstats[8,z] = (suffstats[5,z] * mu_p - x) / (suffstats[5,z] - 1)
    suffstats[9,z] = (suffstats[5,z]* (suffstats[9,z] + mu_p ** 2) - x ** 2) / (suffstats[5,z] - 1) - suffstats[8,z] ** 2
    suffstats[5,z] -= 1
    suffstats[6,z] -= x
    suffstats[7,z] -= np.log(x)
    return suffstats

@numba.jit(nopython=True)
def increment_Mc(x, z, suffstats, eps=1e-10):
    x += eps
    mu_p = suffstats[8,z]
    suffstats[8,z] = (suffstats[5,z] * mu_p + x) / (suffstats[5,z] + 1)
    suffstats[9,z] = (suffstats[5,z] * (suffstats[9,z] + mu_p ** 2) + x ** 2) / (suffstats[5,z] + 1) - suffstats[8,z] ** 2
    suffstats[5,z] += 1
    suffstats[6,z] += x
    suffstats[7,z] += np.log(x)

    return suffstats
 
@numba.jit(nopython=True)
def pdf_k(x, suffstats):
    shape, scale = get_params(suffstats)
    # return scipy.stats.gamma.pdf(x+1e-10, shape, scale=scale)
    values = np.zeros(len(shape))
    for i in range(len(values)):
        values[i] = gamma_pdf(x+1e-10, shape[i], scale[i])
    return values

@numba.jit(nopython=True)
def get_params(suffstats_z):
    updated_suffstats, shape_hat = update_stats(suffstats_z)
    scale_hat = updated_suffstats[4] / updated_suffstats[3]
    # print("shape,scale")
    # print(shape_hat,scale_hat)
    return shape_hat, scale_hat


@numba.jit(nopython=True)
def approximation(shape, ln_a, b, c, d, ln_e, n, max_iter=10,eps=1e-10):
    for _ in range(max_iter):
        # Newton's method
        val = (ln_a + c * (np.log(d + n * shape) - ln_e)) / b
        shape = inv_digamma(val)
        for i in range(len(shape)):
            if shape[i]>0:
                shape[i]=shape[i]
            else:
                shape[i]= eps
    shape = np.nan_to_num(shape, copy=True, nan=eps)
    return shape

@numba.jit(nopython=True)
def inv_digamma(y, max_iter=3):
    x = np.zeros(len(y))
    x[y >= -2.22] = np.exp(y[y >= -2.22]) + 0.5
    x[y < -2.22] = -1 / (y[y < -2.22] + approx_digamma(1))
    # x = np.exp(y) + 0.5 if y >= -2.22 else -1 / (y + digamma(1))
    for _ in range(max_iter):
        for i in range(len(x)):
            x[i] = x[i]-(approx_digamma(x[i]) - y[i]) / approx_digamma(x[i])
    return x


@numba.jit(nopython=True)
def update_stats(suffstats_z):
    updated_suffstats = np.zeros(suffstats_z.shape)
    shape_hat = suffstats_z[8,:] ** 2
    shape_hat /= suffstats_z[9,:]

    ln_a_hat = suffstats_z[0] + suffstats_z[7]
    b_hat = suffstats_z[1] + suffstats_z[5]
    c_hat = suffstats_z[2] + suffstats_z[5]
    e_hat = suffstats_z[4] + suffstats_z[6]
    shape_hat = approximation(shape_hat, ln_a_hat, b_hat, c_hat, suffstats_z[3], np.log(e_hat), suffstats_z[5])
    # print(f"shape_hat:{shape_hat}")
    d_hat = suffstats_z[3] + suffstats_z[5] * shape_hat

    updated_suffstats[0] = ln_a_hat
    updated_suffstats[1] = b_hat
    updated_suffstats[2] = c_hat
    updated_suffstats[3] = d_hat
    updated_suffstats[4] = e_hat
    updated_suffstats[5:10] = suffstats_z[5:10]

    return updated_suffstats, shape_hat
    

@numba.jit(nopython=True)
def approx_digamma(x, eps=1e-10):
#*****************************************************************************80
#
## DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    20 March 2016
#
#  Author:
#
#    Original FORTRAN77 version by Jose Bernardo.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Jose Bernardo,
#    Algorithm AS 103:
#    Psi ( Digamma ) Function,
#    Applied Statistics,
#    Volume 25, Number 3, 1976, pages 315-317.
#
#  Parameters:
#
#    Input, real X, the argument of the digamma function.
#    0 < X.
#
#    Output, real DIGAMMA, the value of the digamma function at X.
#
#    Output, integer IFAULT, error flag.
#    0, no error.
#    1, X <= 0.
#

    #
    #  Check the input.
    #
    # if ( x <= 0.0 ):
    #     value = 0.0
    #     ifault = 1
    #     return value, ifault
    #
    #  Initialize.
    #
    ifault = 0
    value = 0.0
    #
    #  Use approximation for small argument.
    #
    x += eps
    if ( x <= 0.000001 ):
        euler_mascheroni = 0.57721566490153286060
        value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
        return value
    # , ifault
    #
    #  Reduce to DIGAMA(X + N).
    #
    while ( x < 8.5 ):
        value = value - 1.0 / x
        x = x + 1.0
    #
    #  Use Stirling's (actually de Moivre's) expansion.
    #
    r = 1.0 / x
    value = value + np.log ( x ) - 0.5 * r
    r = r * r
    value = value \
        - r * ( 1.0 / 12.0 \
        - r * ( 1.0 / 120.0 \
        - r * ( 1.0 / 252.0 \
        - r * ( 1.0 / 240.0 \
        - r * ( 1.0 / 132.0 ) ) ) ) )
    return value

@numba.jit(nopython=True)
def gamma_pdf(x, alpha, theta):
    #f(x; α, θ) = (x^(α-1) * exp(-x / θ)) / (Γ(α) * θ^α)
    if x < 0 or alpha <= 0 or theta <= 0:
        return 0
    log_gamma_alpha = lgamma(alpha)
    log_pdf = (alpha - 1) * np.log(x) - (x / theta) - log_gamma_alpha - alpha * np.log(theta)
    return np.exp(log_pdf)


def log_multi_beta(param, K=None):
    """
    Logarithm of the multivariate beta function.
    """
    if K is None:
        # param is assumed to be a vector
        return np.sum(gammaln(param)) - gammaln(np.sum(param))
    else:
        # param is assumed to be a scalar
        return K * gammaln(param) - gammaln(K * param)

@numba.jit(nopython=True)
def llh_gamma(N, S, lnS, shape, scale):
    """
    Logarithm of the likelihood for gamma distribution
    """
    if N < 1:
        return 0
    value = (shape-1) * lnS - 1/scale * S - N * shape * np.log(scale) - N * lgamma(shape) 

    value = value if value > N*np.log(PDF_MIN) else N*np.log(PDF_MIN)
    value = value if value < 0  else 0.0
    return value