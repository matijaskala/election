__kernel void add_ballot(int C,
                         __global unsigned long long int *T,
                         __global unsigned char *B)
{
    int k = get_global_id(0)/(C*C);
    int X = get_global_id(0)%(C*C);
    if (B[X%C] > k && B[X/C] > k)
        T[get_global_id(0)]++;
}

__kernel void calculate_scores (int C,
                                __global unsigned long long *T,
                                __global double *S,
                                int A_length,
                                __global int *A)
{
    int k = get_global_id(0)/(C*C);
    int X = get_global_id(0)%(C*C);
    S[get_global_id(0)] = 0.0;
    int x = X%C, y = X/C;
    unsigned long long Vx = T[k*C*C + x*C + x];
    unsigned long long Vy = T[k*C*C + y*C + y];
    if (Vx == 0 || Vy == 0)
        return;
    double Dx = 1.0, Dy = 1.0;
    for (int i = 0; i < A_length; i++) {
        unsigned long long Vi = T[k*C*C + A[i]*C + A[i]];
        if (Vi == 0)
            return;
        Dx = 1.0 * T[k*C*C + x*C + A[i]] / Vi;
        Dy = 1.0 * T[k*C*C + y*C + A[i]] / Vi;
    if (x == y)
        S[get_global_id(0)] = 1.0 / (Dx / Vx);
    else
        S[get_global_id(0)] = 2.0 / (Dx / Vx + Dy / Vy + 1.0 * T[get_global_id(0)] / (Vx * Vy));
    }
}
