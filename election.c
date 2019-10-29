#include <stdint.h>
#include <malloc.h>

typedef struct {
    int candidates;
    uint64_t* data;
    uint64_t votes;
} Tensor;

Tensor* new_tensor(int c) {
    Tensor* t = malloc(sizeof *t);
    t->candidates = c;
    t->data = malloc(5*c*c * sizeof *t->data);
    t->votes = 0;
    return t;
}

void delete_tensor(Tensor* t) {
    free(t->data);
    free(t);
}

void add_ballot(Tensor* t, uint8_t* b) {
    for (int i = 0; i < t->candidates; i++)
        for (int j = 0; j < t->candidates; j++)
            for (uint8_t k = 0; k < 5; k++)
                if (b[i] > k && b[j] > k)
                    t->data[i*t->candidates+j+k*t->candidates*t->candidates]++;
    t->votes++;
}

double get_score(Tensor* t, int idx, int already_elected_length, int* already_elected, uint8_t cutoff) {
    uint64_t idx_votes = t->data[idx*t->candidates+idx+cutoff*t->candidates*t->candidates];
    if (idx_votes == 0)
        return 0;
    double d = 1;
    for (int i = 0; i < already_elected_length; i++) {
        int a = already_elected[i];
        uint64_t a_votes = t->data[a*t->candidates+a+cutoff*t->candidates*t->candidates];
        if (a_votes == 0)
            return 0;
        d += 1.0 * t->data[idx*t->candidates+a+cutoff*t->candidates*t->candidates] / a_votes;
    }
    return idx_votes / d;
}

uint8_t get_cutoff(Tensor* t, double Q, int already_elected_length, int* already_elected) {
    for (uint8_t i = 4; i > 0; i--)
        for (int j = 0; j < already_elected_length; j++)
            if (get_score(t, already_elected[j], already_elected_length, already_elected, i-1) >= Q)
                return i;
    return 0;
}

int get_next_winner(Tensor* t, double Q, int already_elected_length, int* already_elected) {
    int* idx = malloc((t->candidates + 1) * sizeof *idx);
    idx[0] = -1;
    for (uint8_t i = get_cutoff(t, Q, already_elected_length, already_elected); i < 5; i++) {
        if (idx[0] == -1)
            for (int j = 0; j < t->candidates; j++)
                idx[j] = j;
        else if (idx[1] == -1)
            break;
        idx[t->candidates] = -1;
        int* idx2 = malloc((t->candidates + 1) * sizeof *idx);
        int idx2l = 0;
        double val = -1;
        for (int j = 0; idx[j] != -1; j++) {
            double cur = get_score(t, idx[j], already_elected_length, already_elected, i);
            if (cur == val)
                idx2[idx2l++] = idx[j];
            else if (cur > val) {
                val = cur;
                idx2[0] = idx[j];
                idx2l = 1;
            }
        }
        free(idx);
        idx2[idx2l] = -1;
        idx = idx2;
    }
    int r = idx[0];
    free(idx);
    return r;
}

void get_winners(Tensor* t, int W, int* r) {
    for (int i = 0; i < W; i++)
        r[i] = get_next_winner(t, 1.0 * t->votes / W, i, r);
}
