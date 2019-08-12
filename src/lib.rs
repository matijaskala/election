pub struct Matrix {
    candidates: u32,
    data: Vec<u64>,
}

pub fn create_matrix(c: u32) -> Matrix {
    Matrix{candidates: c, data: vec![0; (c*c) as usize]}
}

impl Matrix {
    pub fn add(&mut self, ballot: Vec<u32>) {
        let c = self.candidates as usize;
        assert!(c*c == self.data.len());
        assert!(c == ballot.len());
        for i in 0..c {
            for j in 0..c {
                self.data[i*c+j] += match i == j {
                    true => ballot[i] as u64,
                    false => ballot[i] as u64 * ballot[j] as u64,
                }
            }
        }
    }

    pub fn get(&self, idx: u32, already_elected: Vec<u32>, max_score: u32) -> f64 {
        let c = self.candidates as usize;
        assert!(c*c == self.data.len());
        let mut d = max_score as f64;
        let idx = idx as usize;
        for i in already_elected {
            let i = i as usize;
            d += self.data[idx*c+i] as f64/self.data[i*c+i] as f64;
        }
        self.data[idx*c+idx] as f64/d
    }
}
