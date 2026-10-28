//! Draft tokens for speculative decoding. The coordinator guesses the next few tokens, sends
//! them through the workers with the real one as a single step, and keeps the guesses the head
//! confirms: k guesses cost one trip through the workers instead of k.
//!
//! [`PromptLookup`] needs no model: text often repeats what came before (code, quotes, edits of
//! the prompt), so the tokens that followed an earlier occurrence of the latest n-gram are a
//! good guess. Another drafter (e.g. an MTP head's guesses) plugs in through [`Drafter`].

pub trait Drafter: Send {
    /// Appends up to `max` guesses for the tokens after `history` to `out`.
    fn draft(&mut self, history: &[u32], max: usize, out: &mut Vec<u32>);
}

/// Prompt lookup: finds the most recent earlier occurrence of the last n tokens, trying the
/// longest n first, and proposes the tokens that followed it.
pub struct PromptLookup {
    /// Shortest n-gram worth trusting; shorter matches guess wrong too often to pay off.
    pub min_ngram: usize,
    pub max_ngram: usize,
}

impl Drafter for PromptLookup {
    fn draft(&mut self, history: &[u32], max: usize, out: &mut Vec<u32>) {
        if max == 0 {
            return;
        }
        let len = history.len();
        for n in (self.min_ngram.max(1)..=self.max_ngram.min(len.saturating_sub(1))).rev() {
            let tail = &history[len - n..];
            // Latest match first; it must leave at least one token after it.
            if let Some(start) = (0..len - n).rev().find(|&s| &history[s..s + n] == tail) {
                let follow = &history[start + n..];
                out.extend_from_slice(&follow[..follow.len().min(max)]);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(history: &[u32], max: usize) -> Vec<u32> {
        let mut out = Vec::new();
        PromptLookup { min_ngram: 2, max_ngram: 4 }.draft(history, max, &mut out);
        out
    }

    #[test]
    fn proposes_what_followed_the_latest_match() {
        // "1 2" occurred twice before; the later one was followed by 7 8.
        assert_eq!(draft(&[1, 2, 3, 4, 1, 2, 7, 8, 9, 1, 2], 2), [7, 8]);
        assert_eq!(draft(&[1, 2, 3, 4, 1, 2, 7, 8, 9, 1, 2], 10), [7, 8, 9, 1, 2]);
    }

    #[test]
    fn prefers_the_longest_ngram() {
        // "2 3" last occurred before 9, but "1 2 3" only before 5.
        assert_eq!(draft(&[1, 2, 3, 5, 0, 2, 3, 9, 1, 2, 3], 1), [5]);
    }

    #[test]
    fn nothing_without_a_long_enough_match() {
        assert!(draft(&[1, 2, 3, 4, 5, 3], 4).is_empty());
        assert!(draft(&[1, 2, 1, 2], 0).is_empty());
        assert!(draft(&[], 4).is_empty());
    }
}
