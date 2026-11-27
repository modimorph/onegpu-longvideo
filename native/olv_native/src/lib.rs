//! Optional native cores for onegpu-longvideo. Each function is a drop-in for a pure-Python original
//! and must return the same result (tests in tests/test_native.py compare them).
use pyo3::prelude::*;

/// Levenshtein alignment with the same tie-breaking backtrace as `qa.wer.align` (M, S, D, I).
#[pyfunction]
fn align(r#ref: Vec<String>, hyp: Vec<String>) -> Vec<(String, Option<usize>, Option<usize>)> {
    let (n, k) = (r#ref.len(), hyp.len());
    // map words to ids so the inner loop compares integers
    let mut ids: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut r: Vec<u32> = Vec::with_capacity(n);
    let mut h: Vec<u32> = Vec::with_capacity(k);
    for wd in r#ref.iter() { let l = ids.len() as u32; r.push(*ids.entry(wd.as_str()).or_insert(l)); }
    for wd in hyp.iter() { let l = ids.len() as u32; h.push(*ids.entry(wd.as_str()).or_insert(l)); }
    let w = k + 1;
    let mut d = vec![0u32; (n + 1) * w];
    for i in 0..=n { d[i * w] = i as u32; }
    for j in 0..=k { d[j] = j as u32; }
    for i in 1..=n {
        for j in 1..=k {
            let sub = d[(i - 1) * w + j - 1] + (r[i - 1] != h[j - 1]) as u32;
            let del = d[(i - 1) * w + j] + 1;
            let ins = d[i * w + j - 1] + 1;
            d[i * w + j] = del.min(ins).min(sub);
        }
    }
    let mut ops = Vec::with_capacity(n + k);
    let (mut i, mut j) = (n, k);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && d[i * w + j] == d[(i - 1) * w + j - 1] + (r[i - 1] != h[j - 1]) as u32 {
            ops.push((if r[i - 1] == h[j - 1] { "M" } else { "S" }.to_string(), Some(i - 1), Some(j - 1)));
            i -= 1; j -= 1;
        } else if i > 0 && d[i * w + j] == d[(i - 1) * w + j] + 1 {
            ops.push(("D".to_string(), Some(i - 1), None));
            i -= 1;
        } else {
            ops.push(("I".to_string(), None, Some(j - 1)));
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

/// RMS envelope, same as `voice.envelope`: sqrt(np.convolve(x**2, ones(hop)/hop, "same"))[::hop].
/// Only the kept samples are computed (one direct window sum each), so the work is O(len(x)) instead of O(len(x) * hop).
#[pyfunction]
fn envelope(x: Vec<f64>, hop: usize) -> Vec<f64> {
    let n = x.len();
    let m = hop.max(1);
    if n == 0 { return vec![]; }
    let c = (m - 1) / 2; // np.convolve "same" offset for len(x) >= m
    let inv = 1.0 / m as f64;
    let mut out = Vec::with_capacity(n / m + 1);
    let mut i = 0usize;
    while i < n {
        // full index k = i + c covers x[k-m+1 ..= k]
        let k = i + c;
        let lo = (k + 1).saturating_sub(m);
        let hi = k.min(n - 1);
        let mut s = 0.0f64;
        let mut t = lo;
        while t <= hi { let v = x[t]; s += v * v * inv; t += 1; }
        out.push(s.sqrt());
        i += m;
    }
    out
}

#[pymodule]
fn olv_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(align, m)?)?;
    m.add_function(wrap_pyfunction!(envelope, m)?)?;
    Ok(())
}
