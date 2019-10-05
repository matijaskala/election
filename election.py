#!/usr/bin/env python

class Tensor(object):
    def __init__(self, c):
        self.candidates = c
        self.data = [0] * (5*c*c)
        self.votes = 0
    
    def add(self, ballot):
        c = self.candidates
        for i in range(c):
            for j in range(c):
                for k in range(5):
                    if ballot[i] > k and ballot[j] > k:
                        self.data[i*c+j+k*c*c] += 1
        self.votes += 1
    
    def get(self, idx, already_elected, cutoff):
        c = self.candidates
        idx_votes = self.data[idx*c+idx+cutoff*c*c]
        if idx_votes == 0:
            return 0.0
        d = 1.0
        for i in already_elected:
            i_votes = self.data[i*c+i+cutoff*c*c]
            if i_votes == 0:
                return 0.0
            d += 1.0 * self.data[idx*c+i+cutoff*c*c] / i_votes
        return idx_votes / d
    
    def find_one_winner(self, already_elected, seats):
        idx = []
        q = 1.0 * self.votes / seats
        for cutoff in range(4, -1, -1):
            val = -1.0
            for i in range(self.candidates):
                if i in already_elected:
                    continue
                if cutoff == 0 or self.get(i, already_elected, cutoff - 1) >= q:
                    cur = self.get(i, already_elected, cutoff)
                    if val == cur:
                        idx.append(i)
                    elif val < cur:
                        val = cur
                        idx = [i]
            if idx:
                for i in range(cutoff - 1, -1, -1):
                    if len(idx) == 1:
                        break
                    idx2 = []
                    val = -1.0
                    for j in idx:
                        cur = self.get(j, already_elected, i)
                        if val == cur:
                            idx2.append(j)
                        elif val < cur:
                            val = cur
                            idx2 = [j]
                    idx = idx2
                break
        return idx[0]
    
    def find_winners(self, seats):
        w = []
        for i in range(seats):
            w.append(self.find_one_winner(w, seats))
        return w

t = Tensor(3)
t.add([5,0,3])
t.add([2,1,0])
t.add([1,3,4])
t.add([0,5,0])
w = t.find_winners(2)
