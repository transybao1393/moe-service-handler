//! Chat templates, tokenizing and incremental detokenizing for the HTTP endpoint (CL-2).
//!
//! The two templates are written out here rather than rendered from `tokenizer_config.json`'s
//! Jinja, which would need a template engine for two short formats.

use std::path::Path;

use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatTemplate {
    /// OLMoE-Instruct: `|||IP_ADDRESS|||<|user|>\n…\n<|assistant|>\n`. OLMoE's BOS and EOS is
    /// token 50279, whose text really is `|||IP_ADDRESS|||`.
    Olmoe,
    /// Qwen: `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`.
    Chatml,
}

const OLMOE_BOS_EOS: &str = "|||IP_ADDRESS|||";

#[derive(Clone, Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl ChatTemplate {
    /// Renders `messages` followed by the prompt for the assistant's reply.
    pub fn render(self, messages: &[Message]) -> Result<String, Error> {
        let mut out = String::new();
        if self == ChatTemplate::Olmoe {
            out.push_str(OLMOE_BOS_EOS);
        }
        for (i, m) in messages.iter().enumerate() {
            let last = i + 1 == messages.len();
            let c = &m.content;
            match (self, m.role.as_str()) {
                (ChatTemplate::Olmoe, "system" | "user") => out.push_str(&format!("<|{}|>\n{c}\n", m.role)),
                (ChatTemplate::Olmoe, "assistant") => {
                    out.push_str(&format!("<|assistant|>\n{c}{OLMOE_BOS_EOS}"));
                    if !last {
                        out.push('\n');
                    }
                }
                (ChatTemplate::Chatml, "system" | "user" | "assistant") => {
                    out.push_str(&format!("<|im_start|>{}\n{c}<|im_end|>\n", m.role))
                }
                (_, role) => return Err(Error::Rejected(format!("unsupported message role {role:?}"))),
            }
        }
        out.push_str(match self {
            ChatTemplate::Olmoe => "<|assistant|>\n",
            ChatTemplate::Chatml => "<|im_start|>assistant\n",
        });
        Ok(out)
    }
}

pub struct Codec {
    tokenizer: Tokenizer,
    template: ChatTemplate,
}

impl Codec {
    pub fn load(path: &Path, template: ChatTemplate) -> Result<Codec, Error> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| Error::Config(format!("tokenizer {}: {e}", path.display())))?;
        Ok(Codec { tokenizer, template })
    }

    pub fn from_tokenizer(tokenizer: Tokenizer, template: ChatTemplate) -> Codec {
        Codec { tokenizer, template }
    }

    /// Token ids for a chat, ending with the assistant prompt. The template supplies any BOS.
    pub fn encode_chat(&self, messages: &[Message]) -> Result<Vec<u32>, Error> {
        let text = self.template.render(messages)?;
        let enc = self.tokenizer.encode(text, false).map_err(|e| Error::Protocol(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer.decode(ids, true).map_err(|e| Error::Protocol(format!("detokenize: {e}")))
    }
}

/// Turns a token stream into text pieces. Tokens can end mid-character, so text is decoded
/// from the start each time and released only up to the last complete character.
pub struct Detokenizer<'a> {
    codec: &'a Codec,
    ids: Vec<u32>,
    emitted: usize,
}

impl<'a> Detokenizer<'a> {
    pub fn new(codec: &'a Codec) -> Self {
        Detokenizer { codec, ids: Vec::new(), emitted: 0 }
    }

    /// Adds a token and returns any newly complete text.
    pub fn push(&mut self, token: u32) -> Result<Option<String>, Error> {
        self.ids.push(token);
        let text = self.codec.decode(&self.ids)?;
        if text.ends_with('\u{FFFD}') || text.len() <= self.emitted || !text.is_char_boundary(self.emitted) {
            return Ok(None);
        }
        let piece = text[self.emitted..].to_string();
        self.emitted = text.len();
        Ok(Some(piece))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<Message> {
        pairs.iter().map(|(r, c)| Message { role: r.to_string(), content: c.to_string() }).collect()
    }

    #[test]
    fn olmoe_template_matches_its_jinja() {
        let m = msgs(&[("system", "Be brief."), ("user", "Hi"), ("assistant", "Hello"), ("user", "Bye")]);
        assert_eq!(
            ChatTemplate::Olmoe.render(&m).unwrap(),
            "|||IP_ADDRESS|||<|system|>\nBe brief.\n<|user|>\nHi\n<|assistant|>\nHello|||IP_ADDRESS|||\n<|user|>\nBye\n<|assistant|>\n"
        );
    }

    #[test]
    fn chatml_template() {
        let m = msgs(&[("user", "Hi")]);
        assert_eq!(ChatTemplate::Chatml.render(&m).unwrap(), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn rejects_unknown_roles() {
        assert!(ChatTemplate::Olmoe.render(&msgs(&[("tool", "x")])).is_err());
    }
}

/// Needs OLMoE's tokenizer at `~/models/olmoe-1b-7b/hf/tokenizer.json`.
#[cfg(test)]
mod olmoe_tokenizer {
    use super::*;

    fn codec() -> Codec {
        let home = std::env::var("HOME").unwrap();
        Codec::load(&Path::new(&home).join("models/olmoe-1b-7b/hf/tokenizer.json"), ChatTemplate::Olmoe).unwrap()
    }

    #[test]
    #[ignore = "needs the OLMoE tokenizer on disk"]
    fn chat_encodes_to_streammoe_smoke_prompt() {
        let m = vec![Message { role: "user".into(), content: "What is the capital of France? Answer in one sentence.".into() }];
        let want = [
            50279, 29, 93, 4537, 49651, 187, 1276, 310, 253, 5347, 273, 6181, 32, 37741, 275, 581, 6197,
            15, 187, 29, 93, 515, 5567, 49651, 187,
        ];
        assert_eq!(codec().encode_chat(&m).unwrap(), want);
    }

    #[test]
    #[ignore = "needs the OLMoE tokenizer on disk"]
    fn detokenizes_the_reference_answer_piece_by_piece() {
        let c = codec();
        let mut d = Detokenizer::new(&c);
        let text: String = [510, 5347, 273, 6181, 310, 7785, 15, 50279]
            .into_iter()
            .filter_map(|t| d.push(t).unwrap())
            .collect();
        assert_eq!(text, "The capital of France is Paris.");
    }
}
