//! Expert ownership for the expert split (docs/expert-parallel.md): which worker holds each
//! routed expert of each layer.
//!
//! Every expert has one home worker. The table starts with each layer's experts split evenly
//! and in order (the first share to the first worker, and so on). The controller counts every
//! pick; [`Owners::plan_moves`] then picks a few experts to move so that, layer by layer, the
//! workers run about the same number of picks per token. A move changes the table only once
//! the new owner has the expert ([`Owners::set_owner`]).

/// A planned move of expert `expert` of `layer` from worker `from` to worker `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub layer: u32,
    pub expert: u32,
    pub from: usize,
    pub to: usize,
}

pub struct Owners {
    experts: u32,
    workers: usize,
    /// Worker of each expert, `layer * experts + expert`.
    owner: Vec<u8>,
    /// Picks of each expert since the counts were last halved.
    uses: Vec<u64>,
}

impl Owners {
    /// Each layer's experts split evenly across `workers` (1..=255), in order.
    pub fn even(layers: u32, experts: u32, workers: usize) -> Owners {
        assert!((1..=255).contains(&workers), "{workers} workers");
        let n = layers as usize * experts as usize;
        let owner = (0..n).map(|i| ((i % experts as usize) * workers / experts as usize) as u8).collect();
        Owners { experts, workers, owner, uses: vec![0; n] }
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    fn at(&self, layer: u32, expert: u32) -> usize {
        layer as usize * self.experts as usize + expert as usize
    }

    pub fn owner(&self, layer: u32, expert: u32) -> usize {
        self.owner[self.at(layer, expert)] as usize
    }

    pub fn set_owner(&mut self, layer: u32, expert: u32, worker: usize) {
        let i = self.at(layer, expert);
        self.owner[i] = worker as u8;
    }

    /// Every `(layer, expert)` that `worker` owns, layer by layer.
    pub fn owned_by(&self, worker: usize) -> Vec<(u32, u32)> {
        let e = self.experts as usize;
        (0..self.owner.len()).filter(|&i| self.owner[i] as usize == worker).map(|i| ((i / e) as u32, (i % e) as u32)).collect()
    }

    /// Counts one pick of an expert.
    pub fn count(&mut self, layer: u32, expert: u32) {
        let i = self.at(layer, expert);
        self.uses[i] += 1;
    }

    /// Picks of `layer` each worker ran since the counts were last halved.
    pub fn load(&self, layer: u32) -> Vec<u64> {
        let mut load = vec![0; self.workers];
        let start = self.at(layer, 0);
        for i in start..start + self.experts as usize {
            load[self.owner[i] as usize] += self.uses[i];
        }
        load
    }

    /// Up to `max_moves` moves that even out each layer's load, none leaving a worker with
    /// more than `max_owned` experts in all. Each takes an expert from a layer's busiest worker
    /// to its least busy one, choosing the move that narrows a gap the most first. The table
    /// is unchanged; the counts are halved, so later plans follow newer use.
    pub fn plan_moves(&mut self, max_moves: usize, max_owned: usize) -> Vec<Move> {
        let layers = self.owner.len() / self.experts as usize;
        let mut load: Vec<Vec<u64>> = (0..layers as u32).map(|l| self.load(l)).collect();
        let mut owned = vec![0usize; self.workers];
        for &w in &self.owner {
            owned[w as usize] += 1;
        }
        let mut planned: Vec<Move> = Vec::new();
        while planned.len() < max_moves {
            // (narrowed by, picks moved, move); on a tie the less used expert moves.
            let mut best: Option<(u64, u64, Move)> = None;
            for (l, load) in load.iter().enumerate() {
                let busy = (0..self.workers).max_by_key(|&w| (load[w], std::cmp::Reverse(w))).unwrap();
                let idle = (0..self.workers).min_by_key(|&w| (load[w], w)).unwrap();
                let gap = load[busy] - load[idle];
                if gap < 2 || owned[idle] >= max_owned {
                    continue;
                }
                for e in 0..self.experts {
                    let i = self.at(l as u32, e);
                    let c = self.uses[i];
                    let taken = planned.iter().any(|m| m.layer == l as u32 && m.expert == e);
                    // Moving a pick count c turns the gap into |gap - 2c|: it narrows for 0 < c < gap.
                    if self.owner[i] as usize != busy || c == 0 || c >= gap || taken {
                        continue;
                    }
                    let narrowed = gap - gap.abs_diff(2 * c);
                    if best.is_none_or(|(b, bc, _)| narrowed > b || (narrowed == b && c < bc)) {
                        best = Some((narrowed, c, Move { layer: l as u32, expert: e, from: busy, to: idle }));
                    }
                }
            }
            let Some((_, c, m)) = best else { break };
            load[m.layer as usize][m.from] -= c;
            load[m.layer as usize][m.to] += c;
            owned[m.from] -= 1;
            owned[m.to] += 1;
            planned.push(m);
        }
        for u in &mut self.uses {
            *u /= 2;
        }
        planned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_with_an_even_split_per_layer() {
        let t = Owners::even(3, 16, 2);
        assert_eq!((t.owner(0, 0), t.owner(0, 7), t.owner(0, 8), t.owner(2, 15)), (0, 0, 1, 1));
        let mine = t.owned_by(1);
        assert_eq!(mine.len(), 24);
        assert_eq!((mine[0], mine[8]), ((0, 8), (1, 8)));
        let three = Owners::even(1, 256, 3);
        let shares: Vec<usize> = (0..3).map(|w| three.owned_by(w).len()).collect();
        assert_eq!(shares, [86, 85, 85]);
    }

    #[test]
    fn moves_even_out_the_picks_of_each_layer() {
        let mut t = Owners::even(2, 8, 2);
        // Layer 0: worker 0 runs 10 + 6 + 2 = 18 picks, worker 1 runs 2. Layer 1 is even.
        for (e, n) in [(0, 10), (1, 6), (2, 2), (4, 2)] {
            (0..n).for_each(|_| t.count(0, e));
        }
        for e in [0, 4] {
            t.count(1, e);
        }
        assert_eq!(t.load(0), [18, 2]);
        let moves = t.plan_moves(8, 100);
        // Expert 1 (6 picks) leaves a gap of 4; expert 2 (2 picks) then closes it.
        assert_eq!(
            moves,
            [Move { layer: 0, expert: 1, from: 0, to: 1 }, Move { layer: 0, expert: 2, from: 0, to: 1 }]
        );
        for m in &moves {
            t.set_owner(m.layer, m.expert, m.to);
        }
        // The counts were halved: 5 + 0 against 3 + 1 + 1.
        assert_eq!(t.load(0), [5, 5]);
        assert!(t.plan_moves(8, 100).is_empty(), "balanced");
    }

    #[test]
    fn moves_respect_the_limits() {
        let mut t = Owners::even(1, 8, 2);
        for e in 0..4 {
            (0..4).for_each(|_| t.count(0, e));
        }
        assert_eq!(t.plan_moves(1, 100).len(), 1);
        // Worker 1 already holds its 4 experts and may not hold more.
        assert!(t.plan_moves(8, 4).is_empty());
        // One hot expert cannot be split: moving it would only swap the imbalance.
        let mut t = Owners::even(1, 8, 2);
        (0..10).for_each(|_| t.count(0, 0));
        assert!(t.plan_moves(8, 100).is_empty());
    }
}
