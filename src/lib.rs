pub struct Tensor {
    candidates: u32,
    data: Vec<u64>,
    votes: u64,
}

pub fn new_tensor(c: u32) -> Tensor {
    Tensor{candidates: c, data: vec![0; (5*c*c) as usize], votes: 0}
}

impl Tensor {
    pub fn add_ballot(&mut self, ballot: &[u8]) {
        let c = self.candidates as usize;
        assert!(5*c*c == self.data.len());
        assert!(c == ballot.len());
        for i in 0..c {
            for j in 0..c {
                for k in 0..5 {
                    if ballot[i] > k && ballot[j] > k {
                        self.data[i*c+j+k as usize*c*c] += 1;
                    }
                }
            }
        }
        self.votes += 1;
    }

    fn get_score(&self, idx: usize, already_elected: &[u32], cutoff: usize) -> f64 {
        let c = self.candidates as usize;
        assert!(5*c*c == self.data.len());
        if self.data[idx*c+idx+cutoff*c*c] == 0 { return 0.0 }
        let mut d = 1.0;
        for i in already_elected {
            let i = *i as usize;
            if self.data[i*c+i+cutoff*c*c] == 0 { return 0.0 }
            d += self.data[idx*c+i+cutoff*c*c] as f64/self.data[i*c+i+cutoff*c*c] as f64;
        }
        self.data[idx*c+idx+cutoff*c*c] as f64/d
    }

    fn get_next_winner(&self, already_elected: &[u32], seats: u32) -> u32 {
        let c = self.candidates as usize;
        assert!(5*c*c == self.data.len());
        let mut idx = vec![];
        let q = self.votes as f64/seats as f64;
        for cutoff in (0..5).rev() {
            let mut val = -1.0;
            for i in 0..c {
                let mut skip = false;
                for j in already_elected {
                    if i == *j as usize { skip = true }
                }
                if !skip && (cutoff == 0 || self.get_score(i, already_elected, cutoff - 1) >= q) {
                    let cur = self.get_score(i, already_elected, cutoff);
                    if val == cur { idx.push(i) }
                    else if val < cur {
                        val = cur;
                        idx = vec![i];
                    }
                }
            }
            if idx.len() != 0 {
                for i in cutoff+1..5 {
                    if idx.len() == 1 { break }
                    let mut idx2 = vec![];
                    let mut val = -1.0;
                    for j in idx {
                        let cur = self.get_score(j, already_elected, i);
                        if val == cur { idx2.push(j) }
                        else if val < cur {
                            val = cur;
                            idx2 = vec![j];
                        }
                    }
                    assert!(idx2.len() != 0);
                    idx = idx2;
                }
                break;
            }
        }
        idx[0] as u32
    }

    pub fn get_winners(&self, seats: u32) -> Vec<u32> {
        assert!(seats <= self.candidates);
        let mut w = vec![];
        for _ in 0..seats {
            w.push(self.get_next_winner(&w, seats));
        }
        w
    }
}
