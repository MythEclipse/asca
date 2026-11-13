//! Byte-pair encoding, tanpa dependensi.
//!
//! Model ASCA memakai id token; selama vocab masih 256, "token" sama dengan byte dan
//! model hanya bisa belajar pola pada tingkat huruf. Tokenizer ini membuat vocab ribuan
//! subword sehingga model bisa belajar kata, bukan huruf — syarat pertama sebelum
//! arsitektur ini bisa dipakai untuk teks nyata.
//!
//! Latihannya bekerja pada tabel frekuensi kata, bukan pada seluruh korpus: korpus
//! dipecah menjadi kata unik + hitungannya, lalu merge BPE dijalankan pada kata-kata itu.
//! Ini pola yang sama dengan implementasi BPE produksi dan membuat biaya latihan
//! sebanding dengan jumlah kata unik, bukan jumlah byte.

use std::collections::HashMap;

pub struct Bpe {
    /// id -> byte yang diwakilinya
    vocab: Vec<Vec<u8>>,
    /// pasangan (a, b) -> id hasil merge, untuk encoding cepat
    ranks: HashMap<(u32, u32), u32>,
    /// pasangan yang dipelajari, berurutan
    merges: Vec<(u32, u32, u32)>,
}

/// Pecah teks menjadi kata sesuai batas alami: deret huruf, deret angka, deret tanda
/// baca, dan deret spasi. Satu spasi di depan kata ikut ke dalam kata itu (" the"),
/// supaya model bisa membedakan awal kata dari tengah kata.
fn pretokenize(text: &[u8]) -> Vec<Vec<u8>> {
    #[derive(PartialEq)]
    enum Cls {
        Alpha,
        Digit,
        Other,
        Space,
    }
    fn cls(b: u8) -> Cls {
        if b.is_ascii_alphabetic() {
            Cls::Alpha
        } else if b.is_ascii_digit() {
            Cls::Digit
        } else if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
            Cls::Space
        } else {
            Cls::Other
        }
    }
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut cur_cls = cls(b' ');
    let mut pending_space: Option<u8> = None;
    for &b in text {
        let c = cls(b);
        if c == Cls::Space {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if pending_space.is_some() {
                // spasi kedua dan seterusnya berdiri sendiri
                out.push(vec![pending_space.unwrap()]);
            }
            pending_space = Some(b);
            continue;
        }
        if cur.is_empty() {
            if let Some(sp) = pending_space.take() {
                cur.push(sp);
            }
            cur_cls = c;
        } else if c != cur_cls {
            out.push(std::mem::take(&mut cur));
            cur_cls = c;
        }
        cur.push(b);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if let Some(sp) = pending_space {
        out.push(vec![sp]);
    }
    out
}

impl Bpe {
    /// Latih BPE sampai `vocab_size` id (256 byte dasar + merge).
    pub fn train(text: &[u8], vocab_size: usize) -> Self {
        assert!(vocab_size > 256, "vocab harus lebih dari 256");
        let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let mut ranks: HashMap<(u32, u32), u32> = HashMap::new();
        let mut merges: Vec<(u32, u32, u32)> = Vec::new();

        // tabel frekuensi kata unik
        let mut freq: HashMap<Vec<u8>, u32> = HashMap::new();
        for w in pretokenize(text) {
            *freq.entry(w).or_insert(0) += 1;
        }
        // kata sebagai deret id byte
        let mut words: Vec<(Vec<u32>, u32)> = freq
            .into_iter()
            .map(|(w, c)| (w.iter().map(|&b| b as u32).collect(), c))
            .collect();
        // kata yang lebih pendek dari 2 simbol tidak mungkin di-merge
        words.retain(|(w, _)| w.len() >= 2);

        while vocab.len() < vocab_size {
            // hitung pasangan; iterasi deterministik supaya hasil latihan bisa diulang
            let mut counts: HashMap<(u32, u32), u64> = HashMap::new();
            for (w, c) in words.iter() {
                for i in 0..w.len().saturating_sub(1) {
                    *counts.entry((w[i], w[i + 1])).or_insert(0) += *c as u64;
                }
            }
            if counts.is_empty() {
                break;
            }
            // tie-break: frekuensi tertinggi, lalu pasangan terkecil secara leksikografis
            let mut best: Option<((u32, u32), u64)> = None;
            for (&p, &c) in counts.iter() {
                match best {
                    None => best = Some((p, c)),
                    Some((bp, bc)) => {
                        if c > bc || (c == bc && p < bp) {
                            best = Some((p, c));
                        }
                    }
                }
            }
            let ((a, b), cnt) = best.unwrap();
            if cnt < 2 {
                break;
            }
            let new_id = vocab.len() as u32;
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            merges.push((a, b, new_id));
            ranks.insert((a, b), new_id);
            // terapkan merge ke semua kata
            for (w, _) in words.iter_mut() {
                let mut out = Vec::with_capacity(w.len());
                let mut i = 0;
                while i < w.len() {
                    if i + 1 < w.len() && w[i] == a && w[i + 1] == b {
                        out.push(new_id);
                        i += 2;
                    } else {
                        out.push(w[i]);
                        i += 1;
                    }
                }
                *w = out;
            }
        }

        Bpe {
            vocab,
            ranks,
            merges,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Encode: pecah jadi kata, lalu merge berulang dengan rank terendah (pasangan yang
    /// paling awal dipelajari) sampai tidak ada pasangan yang dikenal.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        for w in pretokenize(text) {
            let mut ids: Vec<u32> = w.iter().map(|&b| b as u32).collect();
            loop {
                if ids.len() < 2 {
                    break;
                }
                let mut best: Option<(usize, u32, u32)> = None;
                for i in 0..ids.len() - 1 {
                    if let Some(&rank) = self.ranks.get(&(ids[i], ids[i + 1])) {
                        match best {
                            None => best = Some((i, rank, 0)),
                            Some((_, br, _)) => {
                                if rank < br {
                                    best = Some((i, rank, 0));
                                }
                            }
                        }
                    }
                }
                match best {
                    None => break,
                    Some((i, _, _)) => {
                        let new_id = self.ranks[&(ids[i], ids[i + 1])];
                        ids.splice(i..i + 2, std::iter::once(new_id));
                    }
                }
            }
            out.extend(ids);
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &i in ids {
            if let Some(b) = self.vocab.get(i as usize) {
                out.extend_from_slice(b);
            }
        }
        out
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"ASCABPE1");
        buf.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for &(a, b, _) in self.merges.iter() {
            buf.extend_from_slice(&a.to_le_bytes());
            buf.extend_from_slice(&b.to_le_bytes());
        }
        std::fs::write(path, buf)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let buf = std::fs::read(path)?;
        if buf.len() < 12 || &buf[..8] != b"ASCABPE1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bukan file BPE ASCA",
            ));
        }
        let n = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let mut ranks = HashMap::new();
        let mut merges = Vec::with_capacity(n);
        for i in 0..n {
            let off = 12 + i * 8;
            let a = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            let b = u32::from_le_bytes(buf[off + 4..off + 8].try_into().unwrap());
            let new_id = vocab.len() as u32;
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            ranks.insert((a, b), new_id);
            merges.push((a, b, new_id));
        }
        Ok(Bpe {
            vocab,
            ranks,
            merges,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_acak() {
        let text = b"the quick brown fox jumps over the lazy dog. the quick brown fox! 12345";
        let bpe = Bpe::train(text, 320);
        let ids = bpe.encode(text);
        assert_eq!(
            bpe.decode(&ids),
            text.to_vec(),
            "decode harus mengembalikan byte asli"
        );
        assert!(ids.len() < text.len(), "encode harus memadatkan");
    }

    #[test]
    fn vocab_terbatas() {
        let text = b"aaaa bbbb aaaa cccc";
        let bpe = Bpe::train(text, 260);
        assert!(bpe.vocab_size() <= 260);
    }

    #[test]
    fn simpan_muat_setara() {
        let text = b"hello world hello there world hello";
        let bpe = Bpe::train(text, 300);
        let p = std::env::temp_dir().join("asca_bpe_test.bin");
        let ps = p.to_str().unwrap();
        bpe.save(ps).unwrap();
        let b2 = Bpe::load(ps).unwrap();
        let ids = bpe.encode(text);
        assert_eq!(
            ids,
            b2.encode(text),
            "hasil encode harus sama setelah simpan/muat"
        );
        assert_eq!(bpe.decode(&ids), b2.decode(&ids));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn teks_kosong() {
        let bpe = Bpe::train(b"abc abc abc", 260);
        assert!(bpe.encode(b"").is_empty());
        assert!(bpe.decode(&[]).is_empty());
    }
}
