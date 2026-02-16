//! RAG pipeline for hardware datasheet retrieval.
//!
//! Phase 4: Index datasheets (markdown/text), retrieve relevant chunks on
//! hardware-related queries, inject into LLM context for board-specific code generation.

use crate::memory::chunker;
use std::path::Path;

/// A chunk of datasheet content with board metadata.
#[derive(Debug, Clone)]
pub struct DatasheetChunk {
    /// Board this chunk applies to (e.g. "nucleo-f401re", "rpi-gpio"), or None for generic.
    pub board: Option<String>,
    /// Source file path (for debugging).
    pub source: String,
    /// Chunk content.
    pub content: String,
}

/// Hardware RAG index — loads and retrieves datasheet chunks.
pub struct HardwareRag {
    chunks: Vec<DatasheetChunk>,
}

fn collect_md_txt_paths(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_md_txt_paths(&path, out);
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str());
            if ext == Some("md") || ext == Some("txt") {
                out.push(path);
            }
        }
    }
}

impl HardwareRag {
    /// Load datasheets from a directory. Expects .md and .txt files.
    /// Filename (without extension) is used as board tag, e.g. `nucleo-f401re.md` → board "nucleo-f401re".
    /// Files in `_generic/` or named `generic.md` have no board filter.
    pub fn load(workspace_dir: &Path, datasheet_dir: &str) -> anyhow::Result<Self> {
        let base = workspace_dir.join(datasheet_dir);
        if !base.exists() || !base.is_dir() {
            return Ok(Self { chunks: Vec::new() });
        }

        let mut paths = Vec::new();
        collect_md_txt_paths(&base, &mut paths);

        let mut chunks = Vec::new();
        let max_tokens = 512;

        for path in paths {
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            if content.trim().is_empty() {
                continue;
            }

            let board = infer_board_from_path(&path, &base);
            let source = path
                .strip_prefix(workspace_dir)
                .unwrap_or(&path)
                .display()
                .to_string();

            for chunk in chunker::chunk_markdown(&content, max_tokens) {
                chunks.push(DatasheetChunk {
                    board: board.clone(),
                    source: source.clone(),
                    content: chunk.content,
                });
            }
        }

        Ok(Self { chunks })
    }

    /// Retrieve chunks relevant to the query and boards.
    /// Uses simple keyword matching (query terms in content) and board filter.
    /// Returns up to `limit` chunks, preferring board-specific matches.
    pub fn retrieve(&self, query: &str, boards: &[String], limit: usize) -> Vec<&DatasheetChunk> {
        if self.chunks.is_empty() || limit == 0 {
            return Vec::new();
        }

        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        let mut scored: Vec<(&DatasheetChunk, f32)> = Vec::new();
        for chunk in &self.chunks {
            let content_lower = chunk.content.to_lowercase();
            let mut score = 0.0f32;

            for term in &query_terms {
                if content_lower.contains(term) {
                    score += 1.0;
                }
            }

            if score > 0.0 {
                // Boost board-specific chunks when they match a configured board
                let board_match = chunk.board.as_ref().map_or(false, |b| boards.contains(b));
                if board_match {
                    score += 2.0;
                }
                scored.push((chunk, score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored.into_iter().map(|(c, _)| c).collect()
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// True if no chunks are indexed.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Infer board tag from file path. `nucleo-f401re.md` → Some("nucleo-f401re").
/// Files in `_generic/` or named `generic.*` → None.
fn infer_board_from_path(path: &Path, base: &Path) -> Option<String> {
    let rel = path.strip_prefix(base).ok()?;
    let parent = rel.parent();
    let stem = path.file_stem()?.to_str()?;

    if stem == "generic" || stem.starts_with("generic_") {
        return None;
    }
    if parent.map_or(false, |p| p.to_str() == Some("_generic")) {
        return None;
    }

    Some(stem.to_string())
}
