#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <utility>

__global__ void add_ballot (uint64_t *T, int *B, int C) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int cutoff = blockIdx.z;
	if (B[x] > cutoff && B[y] > cutoff)
		T[cutoff *C*C + x *C + y]++;
}

__global__ void calculate_scores (double *S, uint64_t *T, int C, int A_length, int *A) {
	int C1 = blockIdx.x;
	int C2 = blockIdx.y;
	int cutoff = blockIdx.z;
	S[cutoff *C*C + C1 *C + C2] = 0.0;
	uint64_t V1 = T[cutoff *C*C + C1 *C + C1];
	uint64_t V2 = T[cutoff *C*C + C2 *C + C2];
	if (V1 == 0 || V2 == 0)
		return;
	double D1 = 1.0;
	double D2 = 1.0;
	for (int i = 0; i < A_length; i++) {
		uint64_t VA = T[cutoff *C*C + A[i] *C + A[i]];
		if (VA == 0)
			return;
		D1 += 1.0 * T[cutoff *C*C + C1 *C + A[i]] / VA;
		D2 += 1.0 * T[cutoff *C*C + C2 *C + A[i]] / VA;
	}
	if (C1 == C2)
		S[cutoff *C*C + C1 *C + C2] = 1.0 / (D1 / V1);
	else
		S[cutoff *C*C + C1 *C + C2] = 2.0 / (D1 / V1 + D2 / V2 + 1.0 * T[cutoff *C*C + C1 *C + C2] / (V1 * V2));
}

class Tensor {
	int C;
	int M;
	uint64_t *D;
	uint64_t V;

	public:

	Tensor (int C, int M) :
		C{C},
		M{M},
		V{0}
	{
		cudaMalloc((void **)&D, M * C * C * sizeof(uint64_t));
		cudaMemset(D, 0, M * C * C * sizeof(uint64_t));
	}

	Tensor (const Tensor&) = delete;

	Tensor (Tensor&&) = default;

	auto vote_count () const { return V; }

	void add_ballot (int *B) {
		::add_ballot<<<1, dim3(C, C, M)>>> (D, B, C);
		V++;
	}

	int get_cutoff (double *S, double Q, int A_length, int *A) const {
		for (int cutoff = M-1; cutoff > 0; cutoff--)
			for (int i = 0; i < C; i++) {
				bool skip = false;
				for (int k = 0; k < A_length; k++)
					if (A[k] == i)
						skip = true;
				if (skip)
					continue;
				if (S[(cutoff-1) *C*C + i *C + i] >= Q)
					return cutoff;
			}
		return 0;
	}

	int get_cutoff (double *S, double Q, int A_length, int *A, bool (*are_valid)(int, int)) const {
		for (int cutoff = M-1; cutoff > 0; cutoff--)
			for (int i = 0; i < C; i++)
				for (int j = 0; j < i; j++) {
					if (!are_valid(i, j))
						continue;
					bool skip = false;
					for (int k = 0; k < A_length; k++)
						if (A[k] == i || A[k] == j)
							skip = true;
					if (skip)
						continue;
					if (S[(cutoff-1) *C*C + i *C + j] >= Q)
						return cutoff;
				}
		return 0;
	}

	int get_next (double Q, int A_length, int *A) const {
		double *_S;
		cudaMalloc((void **)&_S, M * C * C * sizeof(double));
		calculate_scores<<<1,dim3(C, C, M)>>> (_S, D, C, A_length, A);
		double *S = new double[M * C * C];
		cudaMemcpy(S, &_S, M * C * C * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(_S);
		auto idx = new int[C];
		int idx_length = 0;
		for (int cutoff = get_cutoff(S, Q, A_length, A); cutoff < M; cutoff++) {
			if (idx_length == 0)
				for (int i = 0; i < C; i++) {
					bool skip = false;
					for (int k = 0; k < A_length; k++)
						if (A[k] == i)
							skip = true;
					if (skip)
						continue;
					if (cutoff == 0 || S[(cutoff-1) *C*C + i *C + i] >= Q)
						idx[idx_length++] = i;
				}
			else if (idx_length == 1)
				break;
			double val = -1.0;
			auto idx2 = new int[idx_length];
			int idx2_length = 0;
			for (int i = 0; i < idx_length; i++) {
				double cur = S[cutoff *C*C + idx[i] *C + idx[i]];
				if (cur == val)
					idx2[idx2_length++] = idx[i];
				else if (cur > val) {
					val = cur;
					idx2_length = 1;
					idx2[0] = idx[i];
				}
			}
			delete[] idx;
			idx = idx2;
		}
		auto r = idx[0];
		delete[] idx;
		delete[] S;
		return r;
	}

	std::pair<int, int> get_next (double Q, int A_length, int *A, bool (*are_valid)(int, int)) const {
		int *_A;
		double *_S;
		cudaMalloc((void **)&_A, A_length * sizeof(int));
		cudaMemcpy(_A, A, A_length * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void **)&_S, M * C * C * sizeof(double));
		calculate_scores<<<1,dim3(C, C, M)>>> (_S, D, C, A_length, _A);
		double *S = new double[M * C * C];
		cudaMemcpy(S, _S, M * C * C * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(_S);
		cudaFree(_A);
		auto idx = new std::pair<int, int>[C];
		int idx_length = 0;
		for (int cutoff = get_cutoff(S, Q, A_length, A, are_valid); cutoff < M; cutoff++) {
			if (idx_length == 0)
				for (int i = 0; i < C; i++)
					for (int j = 0; j < i; j++) {
						if (!are_valid(i, j))
							continue;
						bool skip = false;
						for (int k = 0; k < A_length; k++)
							if (A[k] == i || A[k] == j)
								skip = true;
						if (skip)
							continue;
						if (cutoff == 0 || S[(cutoff-1) *C*C + i *C + j] >= Q)
							idx[idx_length++] = {i, j};
					}
			else if (idx_length == 1)
				break;
			double val = -1.0;
			auto idx2 = new std::pair<int, int>[idx_length];
			int idx2_length = 0;
			for (int i = 0; i < idx_length; i++) {
				double cur = S[cutoff *C*C + idx[i].first *C + idx[i].second];
				if (cur == val)
					idx2[idx2_length++] = idx[i];
				else if (cur > val) {
					val = cur;
					idx2_length = 1;
					idx2[0] = idx[i];
				}
			}
			delete[] idx;
			idx = idx2;
		}
		auto r = idx[0];
		delete[] idx;
		delete[] S;
		return r;
	}

	int *get_winners (int W, bool (*are_valid)(int, int)) const {
		auto r = new int[W];
		int l = 0;
		while (l < W-1) {
			auto n = get_next(1.0 * vote_count () / W, l, r, are_valid);
			r[l++] = n.first;
			r[l++] = n.second;
		}
		while (l < W-1)
			r[l++] = get_next(1.0 * vote_count () / W, l, r);
		return r;
	}
};
